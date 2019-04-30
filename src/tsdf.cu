#include "tsdf.cuh"


// using namespace CUDASTL;
using std::vector;
using std::default_random_engine;

// CUDA kernel function to integrate a TSDF voxel volume given depth images
namespace ark {


    //hashing device
    vhashing::HashTable<int3, VoxelBlock, BlockHasher, BlockEqual, vhashing::device_memspace>* dev_blockmap_chunks;
    vhashing::HashTable<int3, VoxelBlock, BlockHasher, BlockEqual, vhashing::device_memspace>* dev_blockmap_;
    
    __device__ float3 voxel2world(int3 voxel, float voxel_size){
        float3 p;
        p.x = ((float)voxel.x + 0.5) * voxel_size;
        p.y = ((float)voxel.y + 0.5) * voxel_size;
        p.z = ((float)voxel.z + 0.5) * voxel_size;
        return p;
    }

    __device__ int3 world2voxel(float3 voxelpos, float voxel_size){
        int3 p;
        p.x = floor(voxelpos.x / voxel_size);
        p.y = floor(voxelpos.y / voxel_size);
        p.z = floor(voxelpos.z / voxel_size);
        return p;
    }

    __device__ float3 block2world(int3 idBlock, float block_size){
        float3 p;
        p.x = ((float)idBlock.x + 0.5) * block_size;
        p.y = ((float)idBlock.y + 0.5) * block_size;
        p.z = ((float)idBlock.z + 0.5) * block_size;
        return p;
    }

    __device__ int3 wolrd2block(float3 blockpos, float block_size){
        int3 p;
        p.x = floor(blockpos.x / block_size);
        p.y = floor(blockpos.y / block_size);
        p.z = floor(blockpos.z / block_size);
        return p;
    }

    __host__ bool GpuTsdfGenerator::isPosInCameraFrustum(float x, float y, float z){
        float pt_base_x = param_->vox_origin.x + x;
        float pt_base_y = param_->vox_origin.y + y;
        float pt_base_z = param_->vox_origin.z + z;

        // Convert from base frame camera coordinates to current frame camera coordinates
        float tmp_pt[3] = {0};
        tmp_pt[0] = pt_base_x - c2w_[0 * 4 + 3];
        tmp_pt[1] = pt_base_y - c2w_[1 * 4 + 3];
        tmp_pt[2] = pt_base_z - c2w_[2 * 4 + 3];
        float pt_cam_x =
                c2w_[0 * 4 + 0] * tmp_pt[0] + c2w_[1 * 4 + 0] * tmp_pt[1] + c2w_[2 * 4 + 0] * tmp_pt[2];
        float pt_cam_y =
                c2w_[0 * 4 + 1] * tmp_pt[0] + c2w_[1 * 4 + 1] * tmp_pt[1] + c2w_[2 * 4 + 1] * tmp_pt[2];
        float pt_cam_z =
                c2w_[0 * 4 + 2] * tmp_pt[0] + c2w_[1 * 4 + 2] * tmp_pt[1] + c2w_[2 * 4 + 2] * tmp_pt[2];

        if (pt_cam_z <= 0)
            return false;

        if(pt_cam_z > param_->max_depth)
            return false;

        int pt_pix_x = roundf(K_[0 * 3 + 0] * (pt_cam_x / pt_cam_z) + K_[0 * 3 + 2]);
        int pt_pix_y = roundf(K_[1 * 3 + 1] * (pt_cam_y / pt_cam_z) + K_[1 * 3 + 2]);
        if (pt_pix_x < 0 || pt_pix_x >= im_width_ || pt_pix_y < 0 || pt_pix_y >= im_height_)
            return false;

        return true;
    }

    __host__ float3 GpuTsdfGenerator::getCameraPos(){
        return make_float3(c2w_[0 * 4 + 3],c2w_[1 * 4 + 3],c2w_[2 * 4 + 3]);
    }

    __host__ bool GpuTsdfGenerator::isChunkInCameraFrustum(int x, int y, int z){
        float3 chunk_center = make_float3(((float)x + 0.5) * chunk_size, ((float)y + 0.5) * chunk_size, ((float)z + 0.5) * chunk_size);
        float3 cam_center = getCameraPos();
        // printf("host: isChunkInCameraFrustum  %f %f %f\n", chunk_center.x, chunk_center.y, chunk_center.z);

        float chunkRadius = 0.5f*CHUNK_RADIUS*sqrt(3.0f);
        float3 vec = (cam_center - chunk_center);
        float l = sqrt(vec.x * vec.x + vec.z * vec.z + vec.y * vec.y);
        // printf("\t\t chunkRadius %f \t distance %f\n", chunkRadius, l);

        if(l <= std::abs(chunkRadius))
            return true;
        else
            return false;
    }

    __host__ int GpuTsdfGenerator::chunkGetLinearIdx(int x, int  y, int z){
        int dimx = x + MAX_CHUNK_NUM / 2;
        int dimy = y + MAX_CHUNK_NUM / 2;
        int dimz = z + MAX_CHUNK_NUM / 2;

        return (dimx * BLOCK_PER_CHUNK + dimy) * BLOCK_PER_CHUNK + dimz;
    }

    __host__ int3 GpuTsdfGenerator::world2chunk(float3 pos){
        int3 p;

        p.x = floor(pos.x/chunk_size);
        p.y = floor(pos.y/chunk_size);
        p.z = floor(pos.z/chunk_size);

        return p;
    }

    __global__ void streamInCPU2GPUKernel(int h_inChunkCounter, VoxelBlock* d_inBlock, VoxelBlockPos* d_inBlockPos, vhashing::HashTable<int3, VoxelBlock, BlockHasher, BlockEqual, vhashing::device_memspace> dev_blockmap_chunks){
        const unsigned int bucketId = blockIdx.x * blockDim.x + threadIdx.x;
        // const uint total_vx_p_block = VOXEL_PER_BLOCK * VOXEL_PER_BLOCK * VOXEL_PER_BLOCK;
        if(bucketId < h_inChunkCounter){
            // printf("bucketId: %d of %f\n", bucketId, d_inBlock[bucketId].voxels[0].sdf);

            // printf("bucketId: %d at %d %d %d\n", bucketId, d_inBlockPos[bucketId].pos.x, d_inBlockPos[bucketId].pos.y, d_inBlockPos[bucketId].pos.z);
            dev_blockmap_chunks[d_inBlockPos[bucketId].pos] = d_inBlock[bucketId];
        }
    }

    __host__ void GpuTsdfGenerator::streamInCPU2GPU(float *K, float *c2w, float *depth){

        int block_total = BLOCK_PER_CHUNK * BLOCK_PER_CHUNK * BLOCK_PER_CHUNK;
        h_inChunkCounter = 0;

        printf("HOST: streamInCPU2GPU\n");

        float3 cam_pos = getCameraPos();
        printf("cam_pos = %f %f %f\n", cam_pos.x, cam_pos.y, cam_pos.z);

        int3 camera_chunk = world2chunk(cam_pos);
        printf("MAX_CHUNK_NUM = %d chunk start from %d of size %f\n", MAX_CHUNK_NUM, -MAX_CHUNK_NUM/2, chunk_size);
        printf("camera_chunk = %d %d %d\n", camera_chunk.x, camera_chunk.y, camera_chunk.z);

        int chunk_range_i = ceil(CHUNK_RADIUS/chunk_size); 

        printf("chunk_range_i = %d\n", chunk_range_i);

        int3 chunk_start = make_int3(max(camera_chunk.x - chunk_range_i, - MAX_CHUNK_NUM / 2),
            max(camera_chunk.y - chunk_range_i, - MAX_CHUNK_NUM / 2),
            max(camera_chunk.z - chunk_range_i, - MAX_CHUNK_NUM / 2));

        printf("chunk_start = %d %d %d\n", chunk_start.x, chunk_start.y, chunk_start.z);

        int3 chunk_end = make_int3(min(camera_chunk.x + chunk_range_i, MAX_CHUNK_NUM / 2 - 1),
            min(camera_chunk.y + chunk_range_i, MAX_CHUNK_NUM / 2 - 1),
            min(camera_chunk.z + chunk_range_i, MAX_CHUNK_NUM / 2 - 1));

        printf("chunk_end = %d %d %d\n", chunk_end.x, chunk_end.y, chunk_end.z);


        std::cout<< "before chunk_start ===  clear " <<std::endl;
        clearheap();

        //x y z  real pos idx
        for(int x = chunk_start.x; x < chunk_end.x; x ++){
            for(int y = chunk_start.y; y < chunk_end.y; y ++){
                for(int z = chunk_start.z; z < chunk_end.z; z++){
                    if(isChunkInCameraFrustum(x,y,z)){
                        printf("isChunkInCameraFrustum  %d %d %d\n", x, y, z);
                        int idChunk = chunkGetLinearIdx(x,y,z);
                        printf("idChunk %d %d %d == %d create blocks\n",x, y, z,idChunk);
                        
                        h_chunks[idChunk].create(make_int3(x,y,z));

                        printf("cuda malloc total %d  max :% d\n", h_inChunkCounter, MAX_CPU2GPU_BLOCKS);
                        cudaSafeCall(cudaMemcpy(d_inBlock + h_inChunkCounter,h_chunks[idChunk].blocks, sizeof(VoxelBlock) * block_total, cudaMemcpyHostToDevice));
                        cudaSafeCall(cudaMemcpy(d_inBlockPos + h_inChunkCounter,h_chunks[idChunk].blocksPos, sizeof(VoxelBlockPos) * block_total, cudaMemcpyHostToDevice));
                        h_chunks[idChunk].GPUorCPU = 1;
                        h_inChunkCounter += block_total;
                    }
                }
            }
        }

            // if(isChunkInCameraFrustum(ck)){
            //     // for(int x = 0; x < BLOCK_PER_CHUNK; x ++){
            //     //     for(int y = 0; y < BLOCK_PER_CHUNK; y ++){
            //     //         for(int z = 0; z < BLOCK_PER_CHUNK; z ++){
            //     //             int idBlock = chunkGetLinearIdx(x,y,z);
            //     //             if(ck->blocksPos[idBlock].idx != -1){
            //     //                 h_input_blocks[h_inChunkCounter] = ck->blocksPos[idBlock];
            //     //             }
            //     //         }
            //     //     }
            //     // }
            //     // VoxelBlock* vb = d_inBlock + h_inChunkCounter;
            //     // VoxelBlockPos* vbpos = d_inBlockPos + h_inChunkCounter;
            //     // cudaSafeCall(cudaMalloc(&vb, sizeof(VoxelBlock))); //debug
            //     // cudaSafeCall(cudaMalloc(&vbpos, sizeof(VoxelBlockPos)));  //debug

            //     // cudaSafeCall(cudaMemcpy(d_inBlock + h_inChunkCounter,ck->blocks, sizeof(VoxelBlock) * block_total, cudaMemcpyHostToDevice));
            //     // cudaSafeCall(cudaMemcpy(d_inBlockPos + h_inChunkCounter,ck->blocks, sizeof(VoxelBlockPos) * block_total, cudaMemcpyHostToDevice));
            //     // h_inChunkCounter += block_total;
            // }
        // }

        std::cout<< "TOTAL: h_inChunkCounter === " <<h_inChunkCounter<<std::endl;

        if(h_inChunkCounter > 0){
            const dim3 grid_size((h_inChunkCounter + T_PER_BLOCK * T_PER_BLOCK - 1) / (T_PER_BLOCK * T_PER_BLOCK), 1);
            const dim3 block_size(T_PER_BLOCK * T_PER_BLOCK, 1);

            printf(" streamInCPU2GPUKernel grid %d  block %d \n", (h_inChunkCounter + T_PER_BLOCK * T_PER_BLOCK - 1) / (T_PER_BLOCK * T_PER_BLOCK), T_PER_BLOCK * T_PER_BLOCK);
            
            VoxelBlockPos *h_inBlockPos;
            h_inBlockPos = (VoxelBlockPos *)malloc(sizeof(VoxelBlockPos) * h_inChunkCounter);
            cudaSafeCall(cudaMemcpy(h_inBlockPos, d_inBlockPos, sizeof(VoxelBlockPos) * h_inChunkCounter, cudaMemcpyDeviceToHost));

            VoxelBlock* h_inBlock;
            h_inBlock = (VoxelBlock *)malloc(sizeof(VoxelBlock) * h_inChunkCounter);
            cudaSafeCall(cudaMemcpy(h_inBlock, d_inBlock, sizeof(h_inBlock) * h_inChunkCounter, cudaMemcpyDeviceToHost));

            for(int i = 0; i < h_inChunkCounter; i += 200){
                printf("blockpos[%d] at %d %d %d\n", i, h_inBlockPos[i].pos.x,h_inBlockPos[i].pos.y,h_inBlockPos[i].pos.z);
                printf("block[%d] of %f\n", i, h_inBlock[i].voxels[0].sdf);
            }

            
            vhashing::HashTable<int3, VoxelBlock, BlockHasher, BlockEqual, vhashing::std_memspace>
                bmh(*dev_blockmap_chunks);
            std::cout<<" dev_blockmap_chunks copy "<<std::endl;


            streamInCPU2GPUKernel<<<grid_size, block_size >>> (h_inChunkCounter, d_inBlock, d_inBlockPos, *dev_blockmap_chunks);
            // return;
            cudaSafeCall(cudaDeviceSynchronize()); //debug

            printf("hehehe\n");
            // stream in
            vhashing::HashTable<int3, VoxelBlock, BlockHasher, BlockEqual, vhashing::std_memspace>
            bmhi(*dev_blockmap_chunks);
            printf("yeyeye\n");
            for(int i = 0; i < h_inChunkCounter; i += 200){
                VoxelBlock vb = bmhi[h_inBlockPos[i].pos];
                printf("vb = %f\n", vb.voxels[0].sdf);
            }


        }

        std::cout<< "after streamInCPU2GPUKernel ===  clear " <<std::endl;
        clearheap();
    }

    __host__ void GpuTsdfGenerator::streamOutGPU2CPU(){

    }


    __global__ void IntegrateHashKernel(float *K, float *c2w, float *depth, unsigned char *rgb,
                   int height, int width, MarchingCubeParam *param,  
                   vhashing::HashTable<int3, VoxelBlock, BlockHasher, BlockEqual, vhashing::device_memspace> dev_blockmap_,
                   VoxelBlockPos* d_inBlockPosHeap, unsigned int *d_heapBlockCounter){

        printf("IntegrateHashKernel block %d  thread %d \n", blockIdx.x, threadIdx.x);
        unsigned int idheap = blockIdx.x;
        if(idheap < *d_heapBlockCounter){
            int3 idBlock = d_inBlockPosHeap[idheap].pos;
            VoxelBlock& vb = dev_blockmap_[idBlock];
            Voxel& voxel = vb.voxels[threadIdx.x];

            int VOXEL_PER_BLOCK2 = VOXEL_PER_BLOCK * VOXEL_PER_BLOCK;
            int z = threadIdx.x % VOXEL_PER_BLOCK;
            int y = (threadIdx.x - z) % VOXEL_PER_BLOCK2;
            int x = threadIdx.x / VOXEL_PER_BLOCK2;

            int3 idVoxel = idBlock * VOXEL_PER_BLOCK + make_int3(x,y,z);

            float3 voxelpos = voxel2world(idVoxel, param->vox_size);

            float pt_base_x = param->vox_origin.x + voxelpos.x;
            float pt_base_y = param->vox_origin.y + voxelpos.y;
            float pt_base_z = param->vox_origin.z + voxelpos.z;

            // Convert from base frame camera coordinates to current frame camera coordinates
            float tmp_pt[3] = {0};
            tmp_pt[0] = pt_base_x - c2w[0 * 4 + 3];
            tmp_pt[1] = pt_base_y - c2w[1 * 4 + 3];
            tmp_pt[2] = pt_base_z - c2w[2 * 4 + 3];
            float pt_cam_x =
                    c2w[0 * 4 + 0] * tmp_pt[0] + c2w[1 * 4 + 0] * tmp_pt[1] + c2w[2 * 4 + 0] * tmp_pt[2];
            float pt_cam_y =
                    c2w[0 * 4 + 1] * tmp_pt[0] + c2w[1 * 4 + 1] * tmp_pt[1] + c2w[2 * 4 + 1] * tmp_pt[2];
            float pt_cam_z =
                    c2w[0 * 4 + 2] * tmp_pt[0] + c2w[1 * 4 + 2] * tmp_pt[1] + c2w[2 * 4 + 2] * tmp_pt[2];

            if (pt_cam_z <= 0)
                return;

            int pt_pix_x = roundf(K[0 * 3 + 0] * (pt_cam_x / pt_cam_z) + K[0 * 3 + 2]);
            int pt_pix_y = roundf(K[1 * 3 + 1] * (pt_cam_y / pt_cam_z) + K[1 * 3 + 2]);
            if (pt_pix_x < 0 || pt_pix_x >= width || pt_pix_y < 0 || pt_pix_y >= height)
                return;

            float depth_val = depth[pt_pix_y * width + pt_pix_x];

            if (depth_val <= 0 || depth_val > param->max_depth)
                return;

            float diff = depth_val - pt_cam_z;

            if (diff <= -param->trunc_margin)
                return;

            // Integrate
            int image_idx = pt_pix_y * width + pt_pix_x;
            float dist = fmin(1.0f, diff / param->trunc_margin);
            float weight_old = voxel.weight;
            float weight_new = weight_old + 1.0f;
            voxel.weight = weight_new;
            voxel.sdf = (voxel.sdf * voxel.weight + dist) / weight_new;
            voxel.sdf_color[0] = (voxel.sdf_color[0] * weight_old + rgb[3 * image_idx]) / weight_new;
            voxel.sdf_color[1] = (voxel.sdf_color[1] * weight_old + rgb[3 * image_idx + 1]) / weight_new;
            voxel.sdf_color[2] = (voxel.sdf_color[2] * weight_old + rgb[3 * image_idx + 2]) / weight_new;
        }
    }

    __global__
    void Integrate(float *K, float *c2w, float *depth, unsigned char *rgb,
                   int height, int width, float *TSDF, unsigned char *TSDF_color,
                   float *weight, MarchingCubeParam *param) {

        int pt_grid_z = blockIdx.x;
        int pt_grid_y = threadIdx.x;

        for (int pt_grid_x = 0; pt_grid_x < param->vox_dim.x; ++pt_grid_x) {

            // Convert voxel center from grid coordinates to base frame camera coordinates
            float pt_base_x = param->vox_origin.x + pt_grid_x * param->vox_size;
            float pt_base_y = param->vox_origin.y + pt_grid_y * param->vox_size;
            float pt_base_z = param->vox_origin.z + pt_grid_z * param->vox_size;

            // Convert from base frame camera coordinates to current frame camera coordinates
            float tmp_pt[3] = {0};
            tmp_pt[0] = pt_base_x - c2w[0 * 4 + 3];
            tmp_pt[1] = pt_base_y - c2w[1 * 4 + 3];
            tmp_pt[2] = pt_base_z - c2w[2 * 4 + 3];
            float pt_cam_x =
                    c2w[0 * 4 + 0] * tmp_pt[0] + c2w[1 * 4 + 0] * tmp_pt[1] + c2w[2 * 4 + 0] * tmp_pt[2];
            float pt_cam_y =
                    c2w[0 * 4 + 1] * tmp_pt[0] + c2w[1 * 4 + 1] * tmp_pt[1] + c2w[2 * 4 + 1] * tmp_pt[2];
            float pt_cam_z =
                    c2w[0 * 4 + 2] * tmp_pt[0] + c2w[1 * 4 + 2] * tmp_pt[1] + c2w[2 * 4 + 2] * tmp_pt[2];

            if (pt_cam_z <= 0)
                continue;

            int pt_pix_x = roundf(K[0 * 3 + 0] * (pt_cam_x / pt_cam_z) + K[0 * 3 + 2]);
            int pt_pix_y = roundf(K[1 * 3 + 1] * (pt_cam_y / pt_cam_z) + K[1 * 3 + 2]);
            if (pt_pix_x < 0 || pt_pix_x >= width || pt_pix_y < 0 || pt_pix_y >= height)
                continue;

            float depth_val = depth[pt_pix_y * width + pt_pix_x];

            if (depth_val <= 0 || depth_val > param->max_depth)
                continue;

            float diff = depth_val - pt_cam_z;

            if (diff <= -param->trunc_margin)
                continue;

            // Integrate
            int volume_idx = pt_grid_z * param->vox_dim.y * param->vox_dim.x + pt_grid_y * param->vox_dim.x + pt_grid_x;
            int image_idx = pt_pix_y * width + pt_pix_x;
            float dist = fmin(1.0f, diff / param->trunc_margin);
            float weight_old = weight[volume_idx];
            float weight_new = weight_old + 1.0f;
            weight[volume_idx] = weight_new;
            TSDF[volume_idx] = (TSDF[volume_idx] * weight_old + dist) / weight_new;
            TSDF_color[volume_idx * 3] = (TSDF_color[volume_idx * 3] * weight_old + rgb[3 * image_idx]) / weight_new;
            TSDF_color[volume_idx * 3 + 1] =
                    (TSDF_color[volume_idx * 3 + 1] * weight_old + rgb[3 * image_idx + 1]) / weight_new;
            TSDF_color[volume_idx * 3 + 2] =
                    (TSDF_color[volume_idx * 3 + 2] * weight_old + rgb[3 * image_idx + 2]) / weight_new;
        }
    }

    __global__
    void marchingCubeKernel(float *TSDF, unsigned char *TSDF_color, Triangle *tri, MarchingCubeParam *param) {
        int pt_grid_z = blockIdx.x;
        int pt_grid_y = threadIdx.x;

        int global_index = pt_grid_z * param->vox_dim.x * param->vox_dim.y - pt_grid_y * param->vox_dim.x;

        for (int pt_grid_x = 0; pt_grid_x < param->vox_dim.x; ++pt_grid_x) {
            int index = global_index + pt_grid_x;

            GRIDCELL grid;
            for (int k = 0; k < 8; ++k) {
                int cxi = pt_grid_x + param->idxMap[k][0];
                int cyi = pt_grid_y + param->idxMap[k][1];
                int czi = pt_grid_z + param->idxMap[k][2];
                grid.p[k] = Vertex(cxi, cyi, czi);
                grid.p[k].r = TSDF_color[3 * (czi * param->vox_dim.y * param->vox_dim.z +
                                              cyi * param->vox_dim.z + cxi)];
                grid.p[k].g = TSDF_color[
                        3 * (czi * param->vox_dim.y * param->vox_dim.z + cyi * param->vox_dim.z + cxi) + 1];
                grid.p[k].b = TSDF_color[
                        3 * (czi * param->vox_dim.y * param->vox_dim.z + cyi * param->vox_dim.z + cxi) + 2];
                grid.val[k] = TSDF[czi * param->vox_dim.y * param->vox_dim.z +
                                   cyi * param->vox_dim.z +
                                   cxi];
            }

            int cubeIndex = 0;
            if (grid.val[0] < 0) cubeIndex |= 1;
            if (grid.val[1] < 0) cubeIndex |= 2;
            if (grid.val[2] < 0) cubeIndex |= 4;
            if (grid.val[3] < 0) cubeIndex |= 8;
            if (grid.val[4] < 0) cubeIndex |= 16;
            if (grid.val[5] < 0) cubeIndex |= 32;
            if (grid.val[6] < 0) cubeIndex |= 64;
            if (grid.val[7] < 0) cubeIndex |= 128;

            Vertex vertlist[12];
            if (param->edgeTable[cubeIndex] == 0)
                continue;

            /* Find the vertices where the surface intersects the cube */
            if (param->edgeTable[cubeIndex] & 1)
                vertlist[0] =
                        VertexInterp(0, grid.p[0], grid.p[1], grid.val[0], grid.val[1]);
            if (param->edgeTable[cubeIndex] & 2)
                vertlist[1] =
                        VertexInterp(0, grid.p[1], grid.p[2], grid.val[1], grid.val[2]);
            if (param->edgeTable[cubeIndex] & 4)
                vertlist[2] =
                        VertexInterp(0, grid.p[2], grid.p[3], grid.val[2], grid.val[3]);
            if (param->edgeTable[cubeIndex] & 8)
                vertlist[3] =
                        VertexInterp(0, grid.p[3], grid.p[0], grid.val[3], grid.val[0]);
            if (param->edgeTable[cubeIndex] & 16)
                vertlist[4] =
                        VertexInterp(0, grid.p[4], grid.p[5], grid.val[4], grid.val[5]);
            if (param->edgeTable[cubeIndex] & 32)
                vertlist[5] =
                        VertexInterp(0, grid.p[5], grid.p[6], grid.val[5], grid.val[6]);
            if (param->edgeTable[cubeIndex] & 64)
                vertlist[6] =
                        VertexInterp(0, grid.p[6], grid.p[7], grid.val[6], grid.val[7]);
            if (param->edgeTable[cubeIndex] & 128)
                vertlist[7] =
                        VertexInterp(0, grid.p[7], grid.p[4], grid.val[7], grid.val[4]);
            if (param->edgeTable[cubeIndex] & 256)
                vertlist[8] =
                        VertexInterp(0, grid.p[0], grid.p[4], grid.val[0], grid.val[4]);
            if (param->edgeTable[cubeIndex] & 512)
                vertlist[9] =
                        VertexInterp(0, grid.p[1], grid.p[5], grid.val[1], grid.val[5]);
            if (param->edgeTable[cubeIndex] & 1024)
                vertlist[10] =
                        VertexInterp(0, grid.p[2], grid.p[6], grid.val[2], grid.val[6]);
            if (param->edgeTable[cubeIndex] & 2048)
                vertlist[11] =
                        VertexInterp(0, grid.p[3], grid.p[7], grid.val[3], grid.val[7]);

            int count = 0;
            for (int ti = 0; param->triTable[cubeIndex][ti] != -1; ti += 3) {
                tri[index * 5 + count].p[0] = vertlist[param->triTable[cubeIndex][ti]];
                tri[index * 5 + count].p[1] = vertlist[param->triTable[cubeIndex][ti + 1]];
                tri[index * 5 + count].p[2] = vertlist[param->triTable[cubeIndex][ti + 2]];
                tri[index * 5 + count].valid = true;
                count++;
            }
            assert(count != 0);
        }
    }

    __host__
    void GpuTsdfGenerator::clearheap(){
        unsigned int src = 0;
        unsigned int *d_heap;
        // cudaSafeCall(cudaFree(d_heapBlockCounter));
        cudaSafeCall(cudaMalloc(&d_heap, sizeof(unsigned int)));
        cudaSafeCall(cudaMemcpy(d_heap, &src, sizeof(unsigned int), cudaMemcpyHostToDevice));
        
        cudaSafeCall(cudaMemcpy(d_heapBlockCounter, &src, sizeof(unsigned int), cudaMemcpyHostToDevice));
        std::cout<<" d_heapBlockCounter clear "<<std::endl;
    }

    __host__
    GpuTsdfGenerator::GpuTsdfGenerator(int width, int height, float fx, float fy, float cx, float cy, float max_depth,
                                       float origin_x = -1.5f, float origin_y = -1.5f, float origin_z = 0.5f,
                                       float vox_size = 0.006f, float trunc_m = 0.03f, int vox_dim_x = 500,
                                       int vox_dim_y = 500, int vox_dim_z = 500) {

        std::cout<<" GpuTsdfGenerator init "<<std::endl;

        checkCUDA(__LINE__, cudaGetLastError());

        im_width_ = width;
        im_height_ = height;

        memset(K_, 0.0f, sizeof(float) * 3 * 3);
        K_[0] = fx;
        K_[2] = cx;
        K_[4] = fy;
        K_[5] = cy;
        K_[8] = 1.0f;

        std::cout<<" hashing init "<<std::endl;

        /***                     ----- hashing parameters -----             ***/
        chunk_size = BLOCK_PER_CHUNK * VOXEL_PER_BLOCK * vox_size;

        cudaSafeCall(cudaMalloc(&d_outBlock, sizeof(VoxelBlock) * MAX_CPU2GPU_BLOCKS));
        std::cout<<" d_outBlock init "<<std::endl;
        cudaSafeCall(cudaMalloc(&d_outBlockPos, sizeof(VoxelBlockPos) * MAX_CPU2GPU_BLOCKS));
        std::cout<<" d_outBlockPos init "<<std::endl;
        cudaSafeCall(cudaMalloc(&d_inBlock, sizeof(VoxelBlock) * MAX_CPU2GPU_BLOCKS));
        std::cout<<" d_inBlock init "<<std::endl;
        cudaSafeCall(cudaMalloc(&d_inBlockPos, sizeof(VoxelBlockPos) * MAX_CPU2GPU_BLOCKS));
        std::cout<<" d_inBlockPos init "<<std::endl;
        cudaSafeCall(cudaMalloc(&d_inBlockPosHeap, sizeof(VoxelBlockPos) * MAX_CPU2GPU_BLOCKS));
        std::cout<<" d_inBlockPosHeap init "<<std::endl;
        cudaSafeCall(cudaMalloc(&d_outBlockCounter, sizeof(unsigned int)));
        std::cout<<" d_outBlockCounter init "<<std::endl;
        d_heapBlockCounter = NULL;
        cudaSafeCall(cudaMalloc((void**)&d_heapBlockCounter, sizeof(unsigned int)));
        clearheap();

        printf("Host: HashAssign\n");

        // vhashing::HashTable<int3, VoxelBlock, BlockHasher, BlockEqual, vhashing::device_memspace>
        //     blocks(10000, 2, 19997, int3{999999, 999999, 999999});



        std::cout<<" d_heapBlockCounter init "<<std::endl;
        dev_blockmap_chunks = new vhashing::HashTable<int3, VoxelBlock, BlockHasher, BlockEqual, vhashing::device_memspace>(100001, 4, 400000, int3{999999, 999999, 999999});
        std::cout<<" dev_blockmap_chunks init "<<std::endl;

        dev_blockmap_ = new vhashing::HashTable<int3, VoxelBlock, BlockHasher, BlockEqual, vhashing::device_memspace>(100001, 4, 400000, int3{999999, 999999, 999999});
        std::cout<<" dev_blockmap_ init "<<std::endl;

        h_chunks = new Chunk[MAX_CHUNK_NUM * MAX_CHUNK_NUM * MAX_CHUNK_NUM];        
        std::cout<<" h_chunks init "<<std::endl;
        // /***                            ----- end  -----                    ***/

        // for(int i = 0; i < 9; i ++)
        //     std::cout<<K_[i]<<" - "<<std::endl;
        // std::cout<<std::endl;

        param_ = new MarchingCubeParam();

        param_->vox_origin.x = origin_x;
        param_->vox_origin.y = origin_y;
        param_->vox_origin.z = origin_z;

        param_->vox_dim.x = vox_dim_x;
        param_->vox_dim.y = vox_dim_y;
        param_->vox_dim.z = vox_dim_z;

        param_->max_depth = max_depth;

        param_->min_depth = 0.1;

        param_->vox_size = vox_size;

        param_->trunc_margin = trunc_m;

        param_->total_vox = param_->vox_dim.x * param_->vox_dim.y * param_->vox_dim.z;

        param_->block_size = VOXEL_PER_BLOCK * vox_size;

        param_->im_width = width;
        param_->im_height = height;
        param_->fx = fx;
        param_->fy = fy;
        param_->cx = cx;
        param_->cy = cy;

        // Initialize voxel grid
        TSDF_ = new float[param_->total_vox];
        TSDF_color_ = new unsigned char[3 * param_->total_vox];
        weight_ = new float[param_->total_vox];
        memset(TSDF_, 1.0f, sizeof(float) * param_->total_vox);
        memset(TSDF_color_, 0.0f, 3 * sizeof(unsigned char) * param_->total_vox);
        memset(weight_, 0.0f, sizeof(float) * param_->total_vox);

        tri_ = (Triangle *) malloc(sizeof(Triangle) * param_->total_vox * 5);

        // Load variables to GPU memory
        cudaMalloc(&dev_param_, sizeof(MarchingCubeParam));
        cudaMalloc(&dev_TSDF_, param_->total_vox * sizeof(float));
        cudaMalloc(&dev_TSDF_color_,
                   3 * param_->total_vox * sizeof(unsigned char));
        cudaMalloc(&dev_weight_,
                   param_->total_vox * sizeof(float));
        cudaMalloc(&dev_tri_, sizeof(Triangle) * param_->total_vox * 5);
        checkCUDA(__LINE__, cudaGetLastError());

        cudaMemcpy(dev_param_, param_, sizeof(MarchingCubeParam), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_TSDF_, TSDF_,
                   param_->total_vox * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_TSDF_color_, TSDF_color_,
                   3 * param_->total_vox * sizeof(unsigned char),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(dev_weight_, weight_,
                   param_->total_vox * sizeof(float), cudaMemcpyHostToDevice);
        checkCUDA(__LINE__, cudaGetLastError());

        cudaMalloc(&dev_K_, 3 * 3 * sizeof(float));
        cudaMemcpy(dev_K_, K_, 3 * 3 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMalloc(&dev_c2w_, 4 * 4 * sizeof(float));
        cudaMalloc(&dev_depth_, im_height_ * im_width_ * sizeof(float));
        cudaMalloc(&dev_rgb_, 3 * im_height_ * im_width_ * sizeof(unsigned char));
        checkCUDA(__LINE__, cudaGetLastError());

    }

    __host__
    void inverse_matrix(float* m, float* minv){
        // computes the inverse of a matrix m
        std::cout<<" m = "<<std::endl;

        for(int r=0;r<3;++r){
            for(int c=0;c<3;++c){
                // m[r * 3 + c] = c2w_[r * 4 + c];
                printf("%f\t",m[r * 3 + c]);
            }
            std::cout<<std::endl;
        }

        double det = m[0 * 3 + 0] * (m[1 * 3 + 1] * m[2 * 3 + 2] - m[2 * 3 + 1] * m[1 * 3 + 2]) -
                     m[0 * 3 + 1] * (m[1 * 3 + 0] * m[2 * 3 + 2] - m[1 * 3 + 2] * m[2 * 3 + 0]) +
                     m[0 * 3 + 2] * (m[1 * 3 + 0] * m[2 * 3 + 1] - m[1 * 3 + 1] * m[2 * 3 + 0]);

        double invdet = 1 / det;

        std::cout<<" inverse "<<std::endl;

        // inverse_matrix(m, minv);
        for(int r=0;r<3;++r){
            for(int c=0;c<3;++c){
                printf("%f\t",minv[r * 3 + c]);
            }
            std::cout<<std::endl;
        }

        minv[0 * 3 + 0] = (m[1 * 3 + 1] * m[2 * 3 + 2] - m[2 * 3 + 1] * m[1 * 3 + 2]) * invdet;
        minv[0 * 3 + 1] = (m[0 * 3 + 2] * m[2 * 3 + 1] - m[0 * 3 + 1] * m[2 * 3 + 2]) * invdet;
        minv[0 * 3 + 2] = (m[0 * 3 + 1] * m[1 * 3 + 2] - m[0 * 3 + 2] * m[1 * 3 + 1]) * invdet;
        minv[1 * 3 + 0] = (m[1 * 3 + 2] * m[2 * 3 + 0] - m[1 * 3 + 0] * m[2 * 3 + 2]) * invdet;
        minv[1 * 3 + 1] = (m[0 * 3 + 0] * m[2 * 3 + 2] - m[0 * 3 + 2] * m[2 * 3 + 0]) * invdet;
        minv[1 * 3 + 2] = (m[1 * 3 + 0] * m[0 * 3 + 2] - m[0 * 3 + 0] * m[1 * 3 + 2]) * invdet;
        minv[2 * 3 + 0] = (m[1 * 3 + 0] * m[2 * 3 + 1] - m[2 * 3 + 0] * m[1 * 3 + 1]) * invdet;
        minv[2 * 3 + 1] = (m[2 * 3 + 0] * m[0 * 3 + 1] - m[0 * 3 + 0] * m[2 * 3 + 1]) * invdet;
        minv[2 * 3 + 2] = (m[0 * 3 + 0] * m[1 * 3 + 1] - m[1 * 3 + 0] * m[0 * 3 + 1]) * invdet;
    }


    __host__
    void GpuTsdfGenerator::getLocalGrid(){
        //DDA
        //start from the left bottom of camera plane
        // int startX, startY, startZ;
        // float dX, dY, dZ;


        // int pt_grid_z = blockIdx.x;
        // int pt_grid_y = threadIdx.x;

        // //bottom left of box
        // float pt_pix_x = 0;//(float)im_width_;
        // float pt_pix_y = 0;//(float)im_height_;
        // float pt_cam_bl[3], pt_base_bl[3];
        // frame2cam(pt_pix_x, pt_pix_y, param_->max_depth, pt_cam_bl);
        // cam2base(pt_cam_bl,pt_base_bl);

        // //top right of box
        // pt_pix_x = im_width_;
        // pt_pix_y = im_height_;
        // frame2cam(pt_pix_x, pt_pix_y, param_->max_depth, pt_cam_bl);
        // cam2base(pt_cam_bl,pt_base_bl);

        // float cam_pos[3];
        // float origin[3] = {0,0,0};
        // cam2base(origin, cam_pos);

        // std::cout<<cam_pos[0]<<" "<< cam_pos[1] <<" "<< cam_pos[2]<<std::endl;

        // //top right
        // float pt_pix_x_tr = im_width_ / 2;
        // float pt_pix_y_tr = im_height_ / 2;

        // float pt_cam_z_tr = param_->max_depth;
        // float pt_cam_x_tr = (pt_pix_x_tr - K_[0 * 3 + 2]) * pt_cam_z_tr / K_[0 * 3 + 0];
        // float pt_cam_y_tr = (pt_pix_y_tr - K_[1 * 3 + 2]) * pt_cam_z_tr / K_[1 * 3 + 1];

        // // Convert from current frame camera coordinates to base frame camera coordinates (wolrd)
        // float pt_base_x_tr = pt_cam_x_tr * c2w_[0 * 4 + 0] + pt_cam_y_tr * c2w_[0 * 4 + 1] + pt_cam_z_tr * c2w_[0 * 4 + 2] + c2w_[0 * 4 + 3];
        // float pt_base_y_tr = pt_cam_x_tr * c2w_[1 * 4 + 0] + pt_cam_y_tr * c2w_[1 * 4 + 1] + pt_cam_z_tr * c2w_[1 * 4 + 2] + c2w_[1 * 4 + 3];
        // float pt_base_z_tr = pt_cam_x_tr * c2w_[2 * 4 + 0] + pt_cam_y_tr * c2w_[2 * 4 + 1] + pt_cam_z_tr * c2w_[2 * 4 + 2] + c2w_[2 * 4 + 3];

        // // std::cout<< "btm lft x y z = "<<pt_base_x<<" "<<pt_base_y<<" "<<pt_base_z <<std::endl;
        // // std::cout<< "top rit x y z = "<<pt_base_x_tr<<" "<<pt_base_y_tr<<" "<<pt_base_z_tr <<std::endl;

            

        // // check correctness
        //     // Convert from base frame camera coordinates to current frame camera coordinates
        //     float tmp_pt[3] = {0};
        //     tmp_pt[0] = pt_base[0] - c2w_[0 * 4 + 3];
        //     tmp_pt[1] = pt_base[1] - c2w_[1 * 4 + 3];
        //     tmp_pt[2] = pt_base[2] - c2w_[2 * 4 + 3];
        //     float pt_cam_x =
        //             c2w_[0 * 4 + 0] * tmp_pt[0] + c2w_[1 * 4 + 0] * tmp_pt[1] + c2w_[2 * 4 + 0] * tmp_pt[2];
        //     float pt_cam_y =
        //             c2w_[0 * 4 + 1] * tmp_pt[0] + c2w_[1 * 4 + 1] * tmp_pt[1] + c2w_[2 * 4 + 1] * tmp_pt[2];
        //     float pt_cam_z =
        //             c2w_[0 * 4 + 2] * tmp_pt[0] + c2w_[1 * 4 + 2] * tmp_pt[1] + c2w_[2 * 4 + 2] * tmp_pt[2];

        //      pt_pix_x = roundf(K_[0 * 3 + 0] * (pt_cam_x / pt_cam_z) + K_[0 * 3 + 2]);
        //      pt_pix_y = roundf(K_[1 * 3 + 1] * (pt_cam_y / pt_cam_z) + K_[1 * 3 + 2]);


        // std::cout<< "from x y depth= "<<pt_pix_x<<" "<<pt_pix_y<<" "<<pt_cam_z<<" of "<<- im_width_ / 2 <<" "<<- im_height_ / 2 <<" "<<param_->max_depth<<std::endl;

    }

    __host__
    void GpuTsdfGenerator::processFrame(float *depth, unsigned char *rgb, float *c2w) {
        cudaMemcpy(dev_c2w_, c2w, 4 * 4 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_depth_, depth, im_height_ * im_width_ * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_rgb_, rgb, 3 * im_height_ * im_width_ * sizeof(unsigned char), cudaMemcpyHostToDevice);
        clearheap();
        checkCUDA(__LINE__, cudaGetLastError());

        memcpy(c2w_, c2w, 4 * 4 * sizeof(float));

        std::cout<< "in streamInCPU2GPU ===  clear " <<std::endl;
        clearheap();

        streamInCPU2GPU(dev_K_, dev_c2w_, dev_depth_);
        checkCUDA(__LINE__, cudaGetLastError());

        std::cout<< "out streamInCPU2GPU ===  clear " <<std::endl;
        clearheap();

        HashAssign(dev_depth_, im_height_, im_width_, dev_param_, dev_K_, dev_c2w_);
        // HashAlloc();
        checkCUDA(__LINE__, cudaGetLastError());
        {
            // std::unique_lock<std::mutex> lock(tsdf_mutex_);
            const unsigned int threadsPerBlock = VOXEL_PER_BLOCK*VOXEL_PER_BLOCK*VOXEL_PER_BLOCK;
            unsigned int h_heapBlockCounter;
            cudaSafeCall(cudaMemcpy(&h_heapBlockCounter,d_heapBlockCounter, sizeof(unsigned int), cudaMemcpyDeviceToHost));
            std::cout<<"h_heapBlockCounter == "<<h_heapBlockCounter<<std::endl;

            const dim3 gridSize(h_heapBlockCounter, 1);
            const dim3 blockSize(VOXEL_PER_BLOCK, 1);
            checkCUDA(__LINE__, cudaGetLastError());

            std::cout<<"before IntegrateHashKernel == "<<std::endl;

            IntegrateHashKernel <<< gridSize, blockSize >>> (dev_K_, dev_c2w_, dev_depth_, dev_rgb_, 
                im_height_, im_width_, dev_param_, *dev_blockmap_, d_inBlockPosHeap, d_heapBlockCounter);
        }
        checkCUDA(__LINE__, cudaGetLastError());

        // getLocalGrid();

        {
            std::unique_lock<std::mutex> lock(tsdf_mutex_);
            Integrate << < param_->vox_dim.z, param_->vox_dim.y >> >
                                              (dev_K_, dev_c2w_, dev_depth_, dev_rgb_, im_height_, im_width_, dev_TSDF_, dev_TSDF_color_, dev_weight_, dev_param_);

            checkCUDA(__LINE__, cudaGetLastError());

            cudaMemset(dev_tri_, 0, sizeof(Triangle) * param_->total_vox * 5);
            marchingCubeKernel << < param_->vox_dim.z, param_->vox_dim.y >> >
                                                       (dev_TSDF_, dev_TSDF_color_, dev_tri_, dev_param_);
            checkCUDA(__LINE__, cudaGetLastError());
        }


        {
            std::unique_lock<std::mutex> lock(tri_mutex_);
            cudaMemcpy(tri_, dev_tri_, sizeof(Triangle) * param_->total_vox * 5, cudaMemcpyDeviceToHost);
            checkCUDA(__LINE__, cudaGetLastError());
        }
    }

    __host__ 
    void GpuTsdfGenerator::insert_tri(){



        {
            std::unique_lock<std::mutex> lock(tsdf_mutex_);
            cudaMemcpy(TSDF_, dev_TSDF_,
                       param_->total_vox * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(TSDF_color_, dev_TSDF_color_,
                       3 * param_->total_vox * sizeof(unsigned char), cudaMemcpyDeviceToHost);

            checkCUDA(__LINE__, cudaGetLastError());
        }
    }

    __host__
    void GpuTsdfGenerator::Shutdown() {
        free(tri_);
        cudaFree(dev_TSDF_);
        cudaFree(dev_TSDF_color_);
        cudaFree(dev_weight_);
        cudaFree(dev_K_);
        cudaFree(dev_c2w_);
        cudaFree(dev_depth_);
        cudaFree(dev_rgb_);
        cudaFree(dev_tri_);
        cudaFree(dev_param_);
    }

    __host__ __device__
    Vertex VertexInterp(float isolevel, Vertex p1, Vertex p2, float valp1, float valp2) {
        float mu;
        Vertex p;

        if (fabs(isolevel - valp1) < 0.00001)
            return p1;
        if (fabs(isolevel - valp2) < 0.00001)
            return p2;
        if (fabs(valp1 - valp2) < 0.00001)
            return p1;
        mu = (isolevel - valp1) / (valp2 - valp1);
        p.x = p1.x + mu * (p2.x - p1.x);
        p.y = p1.y + mu * (p2.y - p1.y);
        p.z = p1.z + mu * (p2.z - p1.z);
        p.r = p1.r + mu * (p2.r - p1.r);
        p.g = p1.g + mu * (p2.g - p1.g);
        p.b = p1.b + mu * (p2.b - p1.b);

        return p;
    }

    __host__
    void GpuTsdfGenerator::SaveTSDF(std::string filename) {
        std::unique_lock<std::mutex> lock(tsdf_mutex_);
        // Load TSDF voxel grid from GPU to CPU memory
        cudaMemcpy(TSDF_, dev_TSDF_,
                   param_->total_vox * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(TSDF_color_, dev_TSDF_color_,
                   3 * param_->total_vox * sizeof(unsigned char),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(weight_, dev_weight_,
                   param_->total_vox * sizeof(float), cudaMemcpyDeviceToHost);
        checkCUDA(__LINE__, cudaGetLastError());
        // Save TSDF voxel grid and its parameters to disk as binary file (float array)
        std::cout << "Saving TSDF voxel grid values to disk (tsdf.bin)..." << std::endl;
        std::string voxel_grid_saveto_path = filename;
        std::ofstream outFile(voxel_grid_saveto_path, std::ios::binary | std::ios::out);
        float vox_dim_xf = (float) param_->vox_dim.x;
        float vox_dim_yf = (float) param_->vox_dim.y;
        float vox_dim_zf = (float) param_->vox_dim.z;
        outFile.write((char *) &vox_dim_xf, sizeof(float));
        outFile.write((char *) &vox_dim_yf, sizeof(float));
        outFile.write((char *) &vox_dim_zf, sizeof(float));
        outFile.write((char *) &param_->vox_origin.x, sizeof(float));
        outFile.write((char *) &param_->vox_origin.y, sizeof(float));
        outFile.write((char *) &param_->vox_origin.z, sizeof(float));
        outFile.write((char *) &param_->vox_size, sizeof(float));
        outFile.write((char *) &param_->trunc_margin, sizeof(float));
        for (int i = 0; i < param_->total_vox; ++i)
            outFile.write((char *) &TSDF_[i], sizeof(float));
        for (int i = 0; i < 3 * param_->total_vox; ++i)
            outFile.write((char *) &TSDF_color_[i], sizeof(unsigned char));
        outFile.close();
    }

    __host__
    void GpuTsdfGenerator::SavePLY(std::string filename) {
        {
            std::unique_lock<std::mutex> lock(tsdf_mutex_);
            cudaMemcpy(TSDF_, dev_TSDF_,
                       param_->total_vox * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(TSDF_color_, dev_TSDF_color_,
                       3 * param_->total_vox * sizeof(unsigned char), cudaMemcpyDeviceToHost);

            checkCUDA(__LINE__, cudaGetLastError());
        }
        tsdf2mesh(filename);
    }

    __host__
    void GpuTsdfGenerator::render() {
        std::unique_lock<std::mutex> lock(tri_mutex_);
        for (int i = 0; i < param_->total_vox * 5; ++i) {
            if (!tri_[i].valid)
                continue;
            glBegin(GL_TRIANGLES);
            for (int j = 0; j < 3; ++j) {
                glColor3f(tri_[i].p[j].r / 255.f, tri_[i].p[j].g / 255.f, tri_[i].p[j].b / 255.f);
                glVertex3f(10 * tri_[i].p[j].x * param_->vox_size - 15,
                           -10 * tri_[i].p[j].y * param_->vox_size + 15,
                           -10 * tri_[i].p[j].z * param_->vox_size + 10);
            }
            glEnd();
        }
    }

    __host__
    void GpuTsdfGenerator::tsdf2mesh(std::string outputFileName) {
        std::vector<Face> faces;
        std::vector<Vertex> vertices;

        // std::unordered_map<std::string, int> verticesIdx;
        std::vector<std::list<std::pair<Vertex, int>>> hash_table(param_->total_vox,
                                                                  std::list<std::pair<Vertex, int>>());

        int vertexCount = 0;
        int emptyCount = 0;

        std::cout << "Start saving ply, totalsize: " << param_->total_vox << std::endl;
        for (size_t i = 0; i < param_->total_vox; ++i) {
            int zi = i / (param_->vox_dim.x * param_->vox_dim.y);
            int yi = (i - zi * param_->vox_dim.x * param_->vox_dim.y) / param_->vox_dim.x;
            int xi = i - zi * param_->vox_dim.x * param_->vox_dim.y - yi * param_->vox_dim.x;
            if (xi == param_->vox_dim.x - 1 || yi == param_->vox_dim.y - 1 || zi == param_->vox_dim.z - 1)
                continue;
            GRIDCELL grid;
            std::vector<std::vector<int>> idx_map = {{0, 0, 0},
                                                     {0, 1, 0},
                                                     {1, 1, 0},
                                                     {1, 0, 0},
                                                     {0, 0, 1},
                                                     {0, 1, 1},
                                                     {1, 1, 1},
                                                     {1, 0, 1}};
            for (int k = 0; k < 8; ++k) {
                int cxi = xi + idx_map[k][0];
                int cyi = yi + idx_map[k][1];
                int czi = zi + idx_map[k][2];
                grid.p[k] = Vertex(cxi, cyi, czi);
                grid.p[k].r = TSDF_color_[3 * (czi * param_->vox_dim.y * param_->vox_dim.z +
                                               cyi * param_->vox_dim.z + cxi)];
                grid.p[k].g = TSDF_color_[
                        3 * (czi * param_->vox_dim.y * param_->vox_dim.z + cyi * param_->vox_dim.z + cxi) + 1];
                grid.p[k].b = TSDF_color_[
                        3 * (czi * param_->vox_dim.y * param_->vox_dim.z + cyi * param_->vox_dim.z + cxi) + 2];
                grid.val[k] = TSDF_[czi * param_->vox_dim.y * param_->vox_dim.z + cyi * param_->vox_dim.z +
                                    cxi];
            }

            int cubeIndex = 0;
            if (grid.val[0] < 0) cubeIndex |= 1;
            if (grid.val[1] < 0) cubeIndex |= 2;
            if (grid.val[2] < 0) cubeIndex |= 4;
            if (grid.val[3] < 0) cubeIndex |= 8;
            if (grid.val[4] < 0) cubeIndex |= 16;
            if (grid.val[5] < 0) cubeIndex |= 32;
            if (grid.val[6] < 0) cubeIndex |= 64;
            if (grid.val[7] < 0) cubeIndex |= 128;
            Vertex vertlist[12];
            if (param_->edgeTable[cubeIndex] == 0){
                // std::cout<<"grid "<<xi<<","<<yi<<","<<zi<<","<<" == 0"<<std::endl;
                emptyCount ++;
                continue;
            }

            /* Find the vertices where the surface intersects the cube */
            if (param_->edgeTable[cubeIndex] & 1)
                vertlist[0] =
                        VertexInterp(0, grid.p[0], grid.p[1], grid.val[0], grid.val[1]);
            if (param_->edgeTable[cubeIndex] & 2)
                vertlist[1] =
                        VertexInterp(0, grid.p[1], grid.p[2], grid.val[1], grid.val[2]);
            if (param_->edgeTable[cubeIndex] & 4)
                vertlist[2] =
                        VertexInterp(0, grid.p[2], grid.p[3], grid.val[2], grid.val[3]);
            if (param_->edgeTable[cubeIndex] & 8)
                vertlist[3] =
                        VertexInterp(0, grid.p[3], grid.p[0], grid.val[3], grid.val[0]);
            if (param_->edgeTable[cubeIndex] & 16)
                vertlist[4] =
                        VertexInterp(0, grid.p[4], grid.p[5], grid.val[4], grid.val[5]);
            if (param_->edgeTable[cubeIndex] & 32)
                vertlist[5] =
                        VertexInterp(0, grid.p[5], grid.p[6], grid.val[5], grid.val[6]);
            if (param_->edgeTable[cubeIndex] & 64)
                vertlist[6] =
                        VertexInterp(0, grid.p[6], grid.p[7], grid.val[6], grid.val[7]);
            if (param_->edgeTable[cubeIndex] & 128)
                vertlist[7] =
                        VertexInterp(0, grid.p[7], grid.p[4], grid.val[7], grid.val[4]);
            if (param_->edgeTable[cubeIndex] & 256)
                vertlist[8] =
                        VertexInterp(0, grid.p[0], grid.p[4], grid.val[0], grid.val[4]);
            if (param_->edgeTable[cubeIndex] & 512)
                vertlist[9] =
                        VertexInterp(0, grid.p[1], grid.p[5], grid.val[1], grid.val[5]);
            if (param_->edgeTable[cubeIndex] & 1024)
                vertlist[10] =
                        VertexInterp(0, grid.p[2], grid.p[6], grid.val[2], grid.val[6]);
            if (param_->edgeTable[cubeIndex] & 2048)
                vertlist[11] =
                        VertexInterp(0, grid.p[3], grid.p[7], grid.val[3], grid.val[7]);

            /* Create the triangle */
            for (int ti = 0; param_->triTable[cubeIndex][ti] != -1; ti += 3) {
                Face f;
                Triangle t;
                t.p[0] = vertlist[param_->triTable[cubeIndex][ti]];
                t.p[1] = vertlist[param_->triTable[cubeIndex][ti + 1]];
                t.p[2] = vertlist[param_->triTable[cubeIndex][ti + 2]];

                uint3 grid_size = make_uint3(param_->vox_dim.x, param_->vox_dim.y, param_->vox_dim.z);
                for (int pi = 0; pi < 3; ++pi) {
                    int idx = find_vertex(t.p[pi], grid_size, param_->vox_size, hash_table);
                    if (idx == -1) {
                        insert_vertex(t.p[pi], vertexCount, grid_size, param_->vox_size, hash_table);
                        f.vIdx[pi] = vertexCount++;
                        t.p[pi].x = t.p[pi].x * param_->vox_size + param_->vox_origin.x;
                        t.p[pi].y = t.p[pi].y * param_->vox_size + param_->vox_origin.y;
                        t.p[pi].z = t.p[pi].z * param_->vox_size + param_->vox_origin.z;
                        vertices.push_back(t.p[pi]);
                    } else
                        f.vIdx[pi] = idx;
                }
                faces.push_back(f);
            }
        }

        global_vertex = vertices;
        global_face = faces;
        global_map = hash_table;


        std::cout << vertexCount << std::endl;
        std::ofstream plyFile;
        plyFile.open(outputFileName);
        plyFile << "ply\nformat ascii 1.0\ncomment stanford bunny\nelement vertex ";
        plyFile << vertices.size() << "\n";
        plyFile
                << "property float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\n";
        plyFile << "element face " << faces.size() << "\n";
        plyFile << "property list uchar int vertex_index\nend_header\n";
        for (auto v : vertices) {
            plyFile << v.x << " " << v.y << " " << v.z << " " << (int) v.r << " " << (int) v.g << " " << (int) v.b
                    << "\n";
        }
        for (auto f : faces) {
            plyFile << "3 " << f.vIdx[0] << " " << f.vIdx[1] << " " << f.vIdx[2] << "\n";
        }
        plyFile.close();
        std::cout << "File saved" << std::endl;
        std::cout << "totalsize = "<< param_->total_vox << " empty = "<< emptyCount<< " valid = "<< param_->total_vox - emptyCount << std::endl;
    }

    __host__
    int3 GpuTsdfGenerator::calc_cell_pos(Vertex p, float cell_size) {
        int3 cell_pos;
        cell_pos.x = int(floor(p.x / cell_size));
        cell_pos.y = int(floor(p.y / cell_size));
        cell_pos.z = int(floor(p.z / cell_size));
        return cell_pos;
    }

    __host__
    unsigned int GpuTsdfGenerator::calc_cell_hash(int3 cell_pos, uint3 grid_size) {
        if (cell_pos.x < 0 || cell_pos.x >= (int) grid_size.x
            || cell_pos.y < 0 || cell_pos.y >= (int) grid_size.y
            || cell_pos.z < 0 || cell_pos.z >= (int) grid_size.z)
            return (unsigned int) 0xffffffff;

        cell_pos.x = cell_pos.x & (grid_size.x - 1);
        cell_pos.y = cell_pos.y & (grid_size.y - 1);
        cell_pos.z = cell_pos.z & (grid_size.z - 1);

        return ((unsigned int) (cell_pos.z)) * grid_size.y * grid_size.x
               + ((unsigned int) (cell_pos.y)) * grid_size.x
               + ((unsigned int) (cell_pos.x));
    }

    __host__
    int GpuTsdfGenerator::find_vertex(Vertex p, uint3 grid_size, float cell_size,
                                      std::vector<std::list<std::pair<Vertex, int>>> &hash_table) {
        unsigned int key = calc_cell_hash(calc_cell_pos(p, cell_size), grid_size);
        if (key != 0xffffffff) {
            std::list<std::pair<Vertex, int>> ls = hash_table[key];
            for (auto it = ls.begin(); it != ls.end(); ++it) {
                if ((*it).first.x == p.x && (*it).first.y == p.y && (*it).first.z == p.z) {
                    return (*it).second;
                }
            }
        }
        return -1;
    }

    __host__
    void GpuTsdfGenerator::insert_vertex(Vertex p, int index, uint3 grid_size, float cell_size,
                                         std::vector<std::list<std::pair<Vertex, int>>> &hash_table) {
        unsigned int key = calc_cell_hash(calc_cell_pos(p, cell_size), grid_size);
        if (key != 0xffffffff) {
            std::list<std::pair<Vertex, int>> ls = hash_table[key];
            for (auto it = ls.begin(); it != ls.end(); ++it) {
                if ((*it).first.x == p.x && (*it).first.y == p.y && (*it).first.z == p.z) {
                    (*it).second = index;
                    return;
                }
            }
            ls.push_back(std::make_pair(p, index));
        }
        return;
    }

    __host__
    std::vector<Vertex>* GpuTsdfGenerator::getVertices() {
        return &global_vertex;
    }

    __host__
    std::vector<Face>* GpuTsdfGenerator::getFaces() {
        return &global_face;
    }

    __host__
    std::vector<std::list<std::pair<Vertex, int>>>* GpuTsdfGenerator::getHashMap() {
        return &global_map;
    }

    __host__
    MarchingCubeParam* GpuTsdfGenerator::getMarchingCubeParam() {
        return param_;
    }


    //camera function
    __device__
    void frame2cam(float pt_pix_x, float pt_pix_y, float pt_cam_z, float* pt_cam, float* K_){
        //convert bottom left of frame to current frame camera coordinates (camera)
        pt_cam[2] = pt_cam_z;
        pt_cam[0] = (pt_pix_x - K_[0 * 3 + 2]) * pt_cam_z / K_[0 * 3 + 0];
        pt_cam[1] = (pt_pix_y - K_[1 * 3 + 2]) * pt_cam_z / K_[1 * 3 + 1];

    }

    __device__
    void cam2base(float *pt_cam, float* pt_base, float * c2w_){
        // Convert from current frame camera coordinates to base frame camera coordinates (wolrd)
        pt_base[0] = pt_cam[0] * c2w_[0 * 4 + 0] + pt_cam[1] * c2w_[0 * 4 + 1] + pt_cam[2] * c2w_[0 * 4 + 2] + c2w_[0 * 4 + 3];
        pt_base[1] = pt_cam[0] * c2w_[1 * 4 + 0] + pt_cam[1] * c2w_[1 * 4 + 1] + pt_cam[2] * c2w_[1 * 4 + 2] + c2w_[1 * 4 + 3];
        pt_base[2] = pt_cam[0] * c2w_[2 * 4 + 0] + pt_cam[1] * c2w_[2 * 4 + 1] + pt_cam[2] * c2w_[2 * 4 + 2] + c2w_[2 * 4 + 3];
    }

    __device__
    float3 frame2base(float pt_pix_x, float pt_pix_y, float pt_cam_z, float* K_, float * c2w_){
        float pt_cam[3];
        frame2cam(pt_pix_x, pt_pix_y, pt_cam_z, pt_cam, K_);
        float pt_base[3];
        cam2base(pt_cam, pt_base, c2w_);
        return make_float3(pt_base[0], pt_base[1], pt_base[2]);
    }

    //hashing function

    // __global__
    // void InsertHashGPU(int3 *keys,
    //     VoxelBlock *values,
    //     int n,
    //     vhashing::HashTableBase<int3, VoxelBlock, BlockHasher, BlockEqual> bm) {
    //     int base = blockDim.x * blockIdx.x  +  threadIdx.x;

    //   if (base >= n) {
    //     return;
    //   }
    //   bm[keys[base]] = values[base];
    // }

    bool operator==(const VoxelBlock &a, const VoxelBlock &b) {
        for (int i=0; i<4*4*4; i++) {
            // if(&a.voxels[i] != &b.voxels[i])
            //     return false;
            if(a.voxels[i].sdf != b.voxels[i].sdf) {
                return false;
            }
            // if(a.voxels[i].sdf_color != b.voxels[i].sdf_color){
            //     return false;
            // }
            // if(a.voxels[i].weight != b.voxels[i].weight){
            //     return false;
            // }
        }
        return true;
    }

    // __global__
    // void kernel(int3 *keys,
    //     VoxelBlock *values,
    //     int n,
    //     vhashing::HashTableBase<int3, VoxelBlock, BlockHasher, BlockEqual> bm) {
    //     int base = blockDim.x * blockIdx.x  +  threadIdx.x;

    //   if (base >= n) {
    //     return;
    //   }
    //   bm[keys[base]] = values[base];
    // }

    __device__
    float2 cameraToKinectScreenFloat(const float3& pos, MarchingCubeParam* param)   {
        //return make_float2(pos.x*c_depthCameraParams.fx/pos.z + c_depthCameraParams.mx, c_depthCameraParams.my - pos.y*c_depthCameraParams.fy/pos.z);
        return make_float2(
            pos.x*param->fx/pos.z + param->cx,            
            pos.y*param->fy/pos.z + param->cy);
    }

    __device__
    float cameraToKinectProjZ(float z, MarchingCubeParam* param)    {
        return (z - param->min_depth)/(param->max_depth - param->min_depth);
    }

    __device__
    float3 cameraToKinectProj(const float3& pos, MarchingCubeParam* param) {
        float2 proj = cameraToKinectScreenFloat(pos, param);

        float3 pImage = make_float3(proj.x, proj.y, pos.z);

        pImage.x = (2.0f*pImage.x - (param->im_width- 1.0f))/(param->im_width- 1.0f);
        //pImage.y = (2.0f*pImage.y - (c_depthCameraParams.m_imageHeight-1.0f))/(c_depthCameraParams.m_imageHeight-1.0f);
        pImage.y = ((param->im_height-1.0f) - 2.0f*pImage.y)/(param->im_height-1.0f);
        pImage.z = cameraToKinectProjZ(pImage.z, param);

        return pImage;
    }

    __device__ float3 wolrd2cam(float* c2w_, float3 pos){
                // Convert from base frame camera coordinates to current frame camera coordinates
        float tmp_pt[3] = {0};
        tmp_pt[0] = pos.x - c2w_[0 * 4 + 3];
        tmp_pt[1] = pos.y - c2w_[1 * 4 + 3];
        tmp_pt[2] = pos.z - c2w_[2 * 4 + 3];
        float pt_cam_x =
                c2w_[0 * 4 + 0] * tmp_pt[0] + c2w_[1 * 4 + 0] * tmp_pt[1] + c2w_[2 * 4 + 0] * tmp_pt[2];
        float pt_cam_y =
                c2w_[0 * 4 + 1] * tmp_pt[0] + c2w_[1 * 4 + 1] * tmp_pt[1] + c2w_[2 * 4 + 1] * tmp_pt[2];
        float pt_cam_z =
                c2w_[0 * 4 + 2] * tmp_pt[0] + c2w_[1 * 4 + 2] * tmp_pt[1] + c2w_[2 * 4 + 2] * tmp_pt[2];

        return make_float3(pt_cam_x, pt_cam_y, pt_cam_z);
    }

    __device__ bool isBlockInCameraFrustum(float3 blocks_pos, float* c2w, MarchingCubeParam* param){
        float3 pCamera = wolrd2cam(c2w, blocks_pos);
        float3 pProj = cameraToKinectProj(pCamera, param);
        //pProj *= 1.5f;    //TODO THIS IS A HACK FIX IT :)
        pProj *= 0.95;
        return !(pProj.x < -1.0f || pProj.x > 1.0f || pProj.y < -1.0f || pProj.y > 1.0f || pProj.z < 0.0f || pProj.z > 1.0f); 
    }

    __global__ void HashAssignKernel(float *depth, const unsigned int height, const unsigned int width, 
        MarchingCubeParam *param, float* K, float* c2w, 
        vhashing::HashTable<int3, VoxelBlock, BlockHasher, BlockEqual, vhashing::device_memspace> dev_blockmap_chunks, 
        vhashing::HashTable<int3, VoxelBlock, BlockHasher, BlockEqual, vhashing::device_memspace> dev_blockmap_,
        unsigned int *d_heapBlockCounter, VoxelBlockPos* d_inBlockPosHeap){

        const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
        if(x < width && y < height){
            float d = depth[x * width + y];

            if(d == 0.0f || d == MINF)
                return;

            if(d >= param->max_depth)
                return;

            float t = param->trunc_margin; // debug
            float minDepth = min(param->max_depth, d - t);
            float maxDepth = min(param->max_depth, d + t);
            if(minDepth >= maxDepth)
                return;

            float3 raymin, raymax;
            raymin = frame2base(x,y,minDepth, K, c2w);
            raymax = frame2base(x,y,maxDepth, K, c2w);

            float3 rayDir = normalize(raymax - raymin);
            int3 idCurrentBlock = wolrd2block(raymin, param->block_size);
            int3 idEnd = wolrd2block(raymax, param->block_size);

            float3 step = make_float3(sign(rayDir));
            float3 boundarypos = block2world(idCurrentBlock + make_int3(clamp(step, 0.0, 1.0f)), param->block_size) - 0.5f * param->vox_size;
            float3 tmax = (boundarypos - raymin) / rayDir;
            float3 tDelta = (step * param->vox_size * VOXEL_PER_BLOCK) / rayDir;
            int3 idBound = make_int3(make_float3(idEnd) + step);

            if(rayDir.x == 0.0f || boundarypos.x - raymin.x == 0.0f){ tmax.x = PINF; tDelta.x = PINF;}
            if(rayDir.y == 0.0f || boundarypos.y - raymin.y == 0.0f){ tmax.y = PINF; tDelta.y = PINF;}
            if(rayDir.z == 0.0f || boundarypos.z - raymin.z == 0.0f){ tmax.z = PINF; tDelta.z = PINF;}

            unsigned int iter = 0;
            unsigned int maxLoopIterCount = 1024;

            float3 cam_pos = make_float3(c2w[0 * 4 + 3],c2w[1 * 4 + 3],c2w[2 * 4 + 3]);

            while(iter < maxLoopIterCount){
                float3 blocks_pos = block2world(idCurrentBlock, param->block_size);
                printf("blocks_pos %f %f %f \n", blocks_pos.x, blocks_pos.y, blocks_pos.z);

                if(isBlockInCameraFrustum(blocks_pos, c2w, param)){
                    printf("in camera ++ %d\n", *d_heapBlockCounter);   ///////bug here
                    
                    if(dev_blockmap_chunks.find(idCurrentBlock) != dev_blockmap_chunks.end()){
                        uint addr = atomicAdd(&d_heapBlockCounter[0], 1);
                        dev_blockmap_[idCurrentBlock] = dev_blockmap_chunks[idCurrentBlock];
                        d_inBlockPosHeap[addr].pos = idCurrentBlock;
                    }
                    // else{
                    //     VoxelBlock vb;
                    //     dev_blockmap_[idCurrentBlock] = vb;
                    //     d_inBlockPosHeap[addr].pos = idCurrentBlock;
                    // }
                }else{
                    printf("not in camera\n");
                }

                if(tmax.x < tmax.y && tmax.x < tmax.z){
                    idCurrentBlock.x += step.x;
                    if(idCurrentBlock.x == idBound.x) return;
                    tmax.x += tDelta.x;
                }
                else if(tmax.z < tmax.y)
                {
                    idCurrentBlock.z += step.z;
                    if(idCurrentBlock.z == idBound.z) return;
                    tmax.z += tDelta.z;
                }
                else
                {
                    idCurrentBlock.y += step.y;
                    if(idCurrentBlock.y == idBound.y) return;
                    tmax.y += tDelta.y;
                }  

                iter++;
            }
        }
    }

    __host__ void GpuTsdfGenerator::HashReset(){
        // dev_block_idx.clear();
    }

    __host__ void GpuTsdfGenerator::HashAssign(float *depth, const unsigned int height, const unsigned int width, 
        MarchingCubeParam *param, float* K, float* c2w){

        {
            const dim3 grid_size((im_width_ + T_PER_BLOCK - 1) / T_PER_BLOCK, (im_height_ + T_PER_BLOCK - 1) / T_PER_BLOCK, 1);
            const dim3 block_size(T_PER_BLOCK, T_PER_BLOCK, 1);


            printf("Host: HashAssign\n");

            clearheap();

            unsigned int dst;
            cudaSafeCall(cudaMemcpy(&dst, d_heapBlockCounter, sizeof(unsigned int), cudaMemcpyDeviceToHost));
            printf("Host: HashAssignKernel before d_heapBlockCounter %d\n", dst);

            HashAssignKernel <<< grid_size, block_size >>> (dev_depth_, im_height_, im_width_, dev_param_, dev_K_, dev_c2w_, *dev_blockmap_chunks, *dev_blockmap_, d_heapBlockCounter, d_inBlockPosHeap);
            cudaSafeCall(cudaMemcpy(&dst, d_heapBlockCounter, sizeof(unsigned int), cudaMemcpyDeviceToHost));
            printf("Host: HashAssign d_heapBlockCounter %d\n", dst);
            

        }

        cudaSafeCall(cudaDeviceSynchronize()); //debug
    }
}



