#include "tsdf.cuh"
#include <unordered_map>

// using namespace CUDASTL;
using std::vector;
using std::default_random_engine;

#define DEBUG_WIDTH 32
#define DEBUG_HEIGHT 24
#define DEBUG_X  113
#define DEBUG_Y  26
#define DEBUG_Z  3
#define DDA_STEP 10
// CUDA kernel function to integrate a TSDF voxel volume given depth images
namespace ark {
    // static const int LOCK_HASH = -1;

    //hashing device
    vhashing::HashTable<int3, VoxelBlock, BlockHasher, BlockEqual, vhashing::device_memspace>* dev_blockmap_chunks;
    vhashing::HashTable<int3, VoxelBlock, BlockHasher, BlockEqual, vhashing::device_memspace>* dev_blockmap_;

    static  int countFrame = 0;
    
    __device__ float3 voxel2world(int3 voxel, float voxel_size){
        float3 p;
        p.x = ((float)voxel.x) * voxel_size;
        p.y = ((float)voxel.y) * voxel_size;
        p.z = ((float)voxel.z) * voxel_size;
        return p;
    }

    __device__ int3 world2voxel(float3 voxelpos, float voxel_size){
        int3 p;
        p.x = floorf(voxelpos.x / voxel_size);
        p.y = floorf(voxelpos.y / voxel_size);
        p.z = floorf(voxelpos.z / voxel_size);
        return p;
    }

    __device__ float3 block2world(int3 idBlock, float block_size){
        float3 p;
        p.x = ((float)idBlock.x) * block_size;
        p.y = ((float)idBlock.y) * block_size;
        p.z = ((float)idBlock.z) * block_size;
        return p;
    }

    __device__ int3 wolrd2block(float3 blockpos, float block_size){
        int3 p;
        p.x = floorf(blockpos.x / block_size);
        p.y = floorf(blockpos.y / block_size);
        p.z = floorf(blockpos.z / block_size);
        return p;
    }

    __device__ int3 voxel2block(int3 idVoxel){
        return make_int3(floorf((float)idVoxel.x / (float)VOXEL_PER_BLOCK), floorf((float)idVoxel.y / (float)VOXEL_PER_BLOCK), floorf((float)idVoxel.z / (float)VOXEL_PER_BLOCK));
    }

    __device__ int voxelLinearInBlock(int3 idVoxel, int3 idBlock){
        int3 start_id = make_int3(idBlock.x * VOXEL_PER_BLOCK, idBlock.y * VOXEL_PER_BLOCK, idBlock.z * VOXEL_PER_BLOCK);
                    
        return ((idVoxel.x - start_id.x) * VOXEL_PER_BLOCK + (idVoxel.y - start_id.y))* VOXEL_PER_BLOCK + (idVoxel.z - start_id.z);
    }

    //camera function
    __host__    __device__
    void frame2cam(int* pt_pix, float pt_cam_z, float* pt_cam, float* K_){
        //convert bottom left of frame to current frame camera coordinates (camera)
        pt_cam[2] = pt_cam_z;
        pt_cam[0] = ((float)pt_pix[0] - K_[0 * 3 + 2]) * pt_cam_z / K_[0 * 3 + 0];
        pt_cam[1] = ((float)pt_pix[1] - K_[1 * 3 + 2]) * pt_cam_z / K_[1 * 3 + 1];

    }
    __host__ __device__
    void cam2frame(float* pt_cam, int *pt_pix, float* K){
        pt_pix[0] = roundf(K[0 * 3 + 0] * (pt_cam[0] / pt_cam[2]) + K[0 * 3 + 2]);
        pt_pix[1] = roundf(K[1 * 3 + 1] * (pt_cam[1] / pt_cam[2]) + K[1 * 3 + 2]);
    }

    __host__ __device__
    void base2cam(float* pt_base,float *pt_cam, float * c2w_){
        float tmp_pt[3] = {0};
        tmp_pt[0] = pt_base[0] - c2w_[0 * 4 + 3];
        tmp_pt[1] = pt_base[1] - c2w_[1 * 4 + 3];
        tmp_pt[2] = pt_base[2] - c2w_[2 * 4 + 3];
        pt_cam[0] =
                c2w_[0 * 4 + 0] * tmp_pt[0] + c2w_[1 * 4 + 0] * tmp_pt[1] + c2w_[2 * 4 + 0] * tmp_pt[2];
        pt_cam[1] =
                c2w_[0 * 4 + 1] * tmp_pt[0] + c2w_[1 * 4 + 1] * tmp_pt[1] + c2w_[2 * 4 + 1] * tmp_pt[2];
        pt_cam[2] =
                c2w_[0 * 4 + 2] * tmp_pt[0] + c2w_[1 * 4 + 2] * tmp_pt[1] + c2w_[2 * 4 + 2] * tmp_pt[2];
    }

    __host__  __device__
    void cam2base(float *pt_cam, float* pt_base, float * c2w_){
        // Convert from current frame camera coordinates to base frame camera coordinates (wolrd)
        pt_base[0] = pt_cam[0] * c2w_[0 * 4 + 0] + pt_cam[1] * c2w_[0 * 4 + 1] + pt_cam[2] * c2w_[0 * 4 + 2] + c2w_[0 * 4 + 3];
        pt_base[1] = pt_cam[0] * c2w_[1 * 4 + 0] + pt_cam[1] * c2w_[1 * 4 + 1] + pt_cam[2] * c2w_[1 * 4 + 2] + c2w_[1 * 4 + 3];
        pt_base[2] = pt_cam[0] * c2w_[2 * 4 + 0] + pt_cam[1] * c2w_[2 * 4 + 1] + pt_cam[2] * c2w_[2 * 4 + 2] + c2w_[2 * 4 + 3];

        // //debug herererrere  origin
        // pt_base[0] -= param->vox_origin.x;
        // pt_base[1] -= param->vox_origin.y;
        // pt_base[2] -= param->vox_origin.z;
    }

    __host__  __device__
    float3 frame2base(int pt_pix_x, int pt_pix_y, float pt_cam_z, float* K_, float * c2w_, MarchingCubeParam *param){
        float pt_cam[3];
        int pt_pix[2] = {pt_pix_x, pt_pix_y};
        frame2cam(pt_pix, pt_cam_z, pt_cam, K_);
        float pt_base[3];
        cam2base(pt_cam, pt_base, c2w_);
        return make_float3(pt_base[0], pt_base[1], pt_base[2]);
    }

    __host__ bool GpuTsdfGenerator::isPosInCameraFrustum(float x, float y, float z){
        float pt_base_x = x;// param_->vox_origin.x + x;
        float pt_base_y = y;//param_->vox_origin.y + y;
        float pt_base_z = z;//param_->vox_origin.z + z;

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
        // return make_float3(c2w_[0 * 4 + 3] - param_->vox_origin.x,c2w_[1 * 4 + 3]- param_->vox_origin.y,c2w_[2 * 4 + 3]- param_->vox_origin.z);
        return make_float3(c2w_[0 * 4 + 3],c2w_[1 * 4 + 3],c2w_[2 * 4 + 3]);
    }

    __host__ float3 GpuTsdfGenerator::getFrustumCenter(){
        // float3 getCameraPos();
        int x = im_width_ / 2;
        int y = im_height_ / 2;

        float d = param_->max_depth / 2;

        return frame2base(x,y,d, K_, c2w_, param_);
        // printf("raymin\t%f\t%f\t%f\traymax \t%f\t%f\t%f\torigin\t%f\t%f\t%f\n", 
        //         raymin.x, raymin.y, raymin.z, raymax.x, raymax.y, raymax.z, camcenter.x, camcenter.y, camcenter.z);
    }

    __host__ bool GpuTsdfGenerator::isChunkInCameraFrustum(int x, int y, int z, float3 frustumCenter){

        float3 chunk_center = make_float3(((float)x + 0.5) * chunk_size, ((float)y + 0.5) * chunk_size, ((float)z + 0.5) * chunk_size);
        // float3 cam_center = getCameraPos();
        // printf("host: isChunkInCameraFrustum  %f %f %f\n", chunk_center.x, chunk_center.y, chunk_center.z);

        float chunkRadius = 0.5f*CHUNK_RADIUS*sqrt(3.0f) * 1.1;
        float3 vec = (frustumCenter - chunk_center);
        float l = sqrt(vec.x * vec.x + vec.z * vec.z + vec.y * vec.y);
        // printf("\t\t chunkRadius %f \t distance %f\n", chunkRadius, l);
        // if(x == 4 && y == 3 && z == 1){
        //     printf("4 3 1 chunk_center = (%f,%f,%f) dist = %f chunkRadius = %f\n", chunk_center.x, chunk_center.y, chunk_center.z, l,chunkRadius);
        // }
        // if(x == 2 && y == 1 && z == 1){
        //     printf("2 1 1 chunk_center = (%f,%f,%f) dist = %f chunkRadius = %f\n", chunk_center.x, chunk_center.y, chunk_center.z, l,chunkRadius);
        // }

        if(l <= std::abs(chunkRadius))
            return true;
        else
            return false;
    }

    __host__ int GpuTsdfGenerator::chunkGetLinearIdx(int x, int  y, int z){
        int dimx = x + MAX_CHUNK_NUM / 2;
        int dimy = y + MAX_CHUNK_NUM / 2;
        int dimz = z + MAX_CHUNK_NUM / 2;

        return (dimx * MAX_CHUNK_NUM + dimy) * MAX_CHUNK_NUM + dimz;
    }

    __host__ int3 GpuTsdfGenerator::world2chunk(float3 pos){
        int3 p;

        p.x = floor(pos.x/chunk_size);
        p.y = floor(pos.y/chunk_size);
        p.z = floor(pos.z/chunk_size);


        return p;
    }

    __global__ void streamInCPU2GPUKernel(int h_inChunkCounter, VoxelBlock* d_inBlock, VoxelBlockPos* d_inBlockPos, vhashing::HashTableBase<int3, VoxelBlock, BlockHasher, BlockEqual> dev_blockmap_chunks){
        const unsigned int bucketId = blockIdx.x * blockDim.x + threadIdx.x;
        // const uint total_vx_p_block = VOXEL_PER_BLOCK * VOXEL_PER_BLOCK * VOXEL_PER_BLOCK;
        if(bucketId < h_inChunkCounter){
            int3 pos = d_inBlockPos[bucketId].pos;
            // if(dev_blockmap_chunks.find(pos) == dev_blockmap_chunks.end())
                dev_blockmap_chunks[pos] = d_inBlock[bucketId];
        }
    }


    __global__ void checkStreamInKernel(vhashing::HashTableBase<int3, VoxelBlock, BlockHasher, BlockEqual> dev_blockmap_chunks){
        const unsigned int bucketId = blockIdx.x * blockDim.x + threadIdx.x;

            if(bucketId == 0){
                if(dev_blockmap_chunks.find(make_int3(999,999,999)) != dev_blockmap_chunks.end())
                    printf("insert successfully  (999,999,999)\n");
                else
                    printf("not 999 999 999\n");

                VoxelBlock vb;
                dev_blockmap_chunks[make_int3(999,999,999)] = vb;

                if(dev_blockmap_chunks.find(make_int3(999,999,999)) != dev_blockmap_chunks.end())
                    printf("insert successfully  (999,999,999)\n");
                else
                    printf("not 999 999 999\n");
            }
    }


    __host__ int3 blockLinear2Int3(int idLinearBlock){
        int z = idLinearBlock % BLOCK_PER_CHUNK;
        int y = floor((idLinearBlock % (BLOCK_PER_CHUNK * BLOCK_PER_CHUNK))/BLOCK_PER_CHUNK);
        int x = floor(idLinearBlock / (BLOCK_PER_CHUNK * BLOCK_PER_CHUNK));
        return make_int3(x,y,z);
    }


    __host__ int3 chunk2block(int3 idChunk, int idLinearBlock){
        int3 id = blockLinear2Int3(idLinearBlock);

        int x = idChunk.x * BLOCK_PER_CHUNK + id.x;
        int y = idChunk.y * BLOCK_PER_CHUNK + id.y;
        int z = idChunk.z * BLOCK_PER_CHUNK + id.z;
        return make_int3(x,y,z);
    }

    __host__ __device__ int3 block2chunk(int3 idBlock){
        return make_int3(floor((float)idBlock.x/(float)BLOCK_PER_CHUNK),
            floor((float)idBlock.y/(float)BLOCK_PER_CHUNK),
            floor((float)idBlock.z/(float)BLOCK_PER_CHUNK));
    }


    __host__ int blockPos2Linear(int3 blocksPos){
        int3 chunkpos = block2chunk(blocksPos);
        int3 insideId = blocksPos - chunkpos * make_int3(BLOCK_PER_CHUNK);
        return (insideId.x * BLOCK_PER_CHUNK + insideId.y) * BLOCK_PER_CHUNK + insideId.z;
    }



    __host__ float3 chunk2world(int3 idChunk, float chunk_size){
        return make_float3(((float)(idChunk.x) + 0.5)* chunk_size,
            ((float)(idChunk.y) + 0.5)* chunk_size,
            ((float)(idChunk.z) + 0.5)* chunk_size);
    }

    __host__ void GpuTsdfGenerator::streamInCPU2GPU(float *K, float *c2w, float *depth){

        int block_total = BLOCK_PER_CHUNK * BLOCK_PER_CHUNK * BLOCK_PER_CHUNK;
        h_inChunkCounter = 0;

        // printf("HOST: streamInCPU2GPU\n");

        float3 cam_pos = getCameraPos();
        // printf("cam_pos = %f %f %f\n", cam_pos.x, cam_pos.y, cam_pos.z);
        int3 camera_ck = world2chunk(cam_pos);
        // printf("camera_chunk = %d %d %d \n", camera_ck.x, camera_ck.y, camera_ck.z);
        
        float3 frustumCenter = getFrustumCenter();
        // printf("frustumCenter = %f %f %f\n", frustumCenter.x, frustumCenter.y, frustumCenter.z);

        int3 camera_chunk = world2chunk(frustumCenter);

        float3 chunk_center = chunk2world(camera_chunk, chunk_size);
        // printf("MAX_CHUNK_NUM = %d chunk start from %d of size %f\n", MAX_CHUNK_NUM, -MAX_CHUNK_NUM/2, chunk_size);
        // // printf("frustum_chunk = %d %d %d  pos = %f %f %f block = %d %d %d \n", camera_chunk.x, camera_chunk.y, camera_chunk.z,
        //     chunk_center.x, chunk_center.y, chunk_center.z,
        //     camera_chunk.x * BLOCK_PER_CHUNK, camera_chunk.y * BLOCK_PER_CHUNK, camera_chunk.z * BLOCK_PER_CHUNK);

        int chunk_range_i = ceil(CHUNK_RADIUS/chunk_size); 

        // printf("chunk_range_i = %d\n", chunk_range_i);

        int3 chunk_start = make_int3(max(camera_chunk.x - chunk_range_i, - MAX_CHUNK_NUM / 2),
            max(camera_chunk.y - chunk_range_i, - MAX_CHUNK_NUM / 2),
            max(camera_chunk.z - chunk_range_i, - MAX_CHUNK_NUM / 2));

        // printf("chunk_start = %d %d %d\n", chunk_start.x, chunk_start.y, chunk_start.z);

        int3 chunk_end = make_int3(min(camera_chunk.x + chunk_range_i, MAX_CHUNK_NUM / 2 - 1),
            min(camera_chunk.y + chunk_range_i, MAX_CHUNK_NUM / 2 - 1),
            min(camera_chunk.z + chunk_range_i, MAX_CHUNK_NUM / 2 - 1));

        // printf("chunk_end = %d %d %d\n", chunk_end.x, chunk_end.y, chunk_end.z);

        std::unique_lock<std::mutex> lock(chunk_mutex_);

        //x y z  real pos idx
        for(int x = chunk_start.x; x <= chunk_end.x; x ++){   //should reach end   ======
            for(int y = chunk_start.y; y <= chunk_end.y; y ++){
                for(int z = chunk_start.z; z <= chunk_end.z; z++){
                    int idChunk = chunkGetLinearIdx(x,y,z);

                    if(isChunkInCameraFrustum(x,y,z, frustumCenter)){

                        // int3 stt = chunk2block(make_int3(x,y,z),0);
                        // int3 endd = chunk2block(make_int3(x,y,z),block_total - 1);

                        if(h_chunks[idChunk].blocks == nullptr){
                            h_chunks[idChunk].create(make_int3(x,y,z));
                        }

                        // if(x == 2 && y == 1 && z == 0){
                            // printf("idChunk\t%d,%d,%d\t%d\tof\t%d,%d,%d\tto\t%d,%d,%d\n",x, y, z,idChunk,stt.x,stt.y,stt.z, endd.x, endd.y, endd.z);
                            // if(stt.x <= 17 && endd.x >= 17 &&
                            //     stt.y <= 16 && endd.y >= 16 &&
                            //     stt.z <= 4 && endd.z >= 4)
                            //     printf("successfully 17 16 4\n");
                            // if(stt.x <= 24 && endd.x >= 24 &&
                            //     stt.y <= 9 && endd.y >= 9 &&
                            //     stt.z <= -1 && endd.z >= -1)
                            //     printf("successfully 24 9 -1\n");

                            // if(stt.x <= 25 && endd.x >= 25 &&
                            //     stt.y <= 22 && endd.y >= 22 &&
                            //     stt.z <= 9 && endd.z >= 9)
                            //     printf("successfully 25 22 9\n");

                            // printf("instore %d %d %d == %d of blocks from %d %d %d to %d %d %d \n",x, y, z,
                            //     idChunk,
                            //     h_chunks[idChunk].blocksPos[0].pos.x,h_chunks[idChunk].blocksPos[0].pos.y, h_chunks[idChunk].blocksPos[0].pos.z,
                            //     h_chunks[idChunk].blocksPos[block_total-1].pos.x,
                            //     h_chunks[idChunk].blocksPos[block_total-1].pos.y, 
                            //     h_chunks[idChunk].blocksPos[block_total-1].pos.z);
                        // }
                        

                        // printf("cuda malloc total %d  max :% d\n", h_inChunkCounter, MAX_CPU2GPU_BLOCKS);
                        cudaSafeCall(cudaMemcpy(d_inBlock + h_inChunkCounter,h_chunks[idChunk].blocks, sizeof(VoxelBlock) * block_total, cudaMemcpyHostToDevice));
                        cudaSafeCall(cudaMemcpy(d_inBlockPos + h_inChunkCounter,h_chunks[idChunk].blocksPos, sizeof(VoxelBlockPos) * block_total, cudaMemcpyHostToDevice));
                        // h_chunks[idChunk].isOnGPU = 1;
                        h_inChunkCounter += block_total;
                    }else if(h_chunks[idChunk].isOccupied == false){
                        h_chunks[idChunk].release();
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

        // std::cout<< "TOTAL: h_inChunkCounter === " <<h_inChunkCounter<<std::endl;

        if(h_inChunkCounter > 0){
            const dim3 grid_size((h_inChunkCounter + T_PER_BLOCK * T_PER_BLOCK - 1) / (T_PER_BLOCK * T_PER_BLOCK), 1);
            const dim3 block_size(T_PER_BLOCK * T_PER_BLOCK, 1);

            // printf(" streamInCPU2GPUKernel grid %d  block %d \n", (h_inChunkCounter + T_PER_BLOCK * T_PER_BLOCK - 1) / (T_PER_BLOCK * T_PER_BLOCK), T_PER_BLOCK * T_PER_BLOCK);
            
            // VoxelBlockPos *h_inBlockPos;
            // h_inBlockPos = (VoxelBlockPos *)malloc(sizeof(VoxelBlockPos) * h_inChunkCounter);
            // cudaSafeCall(cudaMemcpy(h_inBlockPos, d_inBlockPos, sizeof(VoxelBlockPos) * h_inChunkCounter, cudaMemcpyDeviceToHost));

            // VoxelBlock* h_inBlock;
            // h_inBlock = (VoxelBlock *)malloc(sizeof(VoxelBlock) * h_inChunkCounter);
            // cudaSafeCall(cudaMemcpy(h_inBlock, d_inBlock, sizeof(h_inBlock) * h_inChunkCounter, cudaMemcpyDeviceToHost));

            // for(int i = 0; i < h_inChunkCounter; i += 1000){
            //     printf("blockpos[%d] at %d %d %d\n", i, h_inBlockPos[i].pos.x,h_inBlockPos[i].pos.y,h_inBlockPos[i].pos.z);
            //     printf("block[%d] of %f\n", i, h_inBlock[i].voxels[0].sdf);
            // }

            streamInCPU2GPUKernel<<<grid_size, block_size >>> (h_inChunkCounter, d_inBlock, d_inBlockPos, *dev_blockmap_chunks);
            // return;
            // cudaSafeCall(cudaDeviceSynchronize()); //debug

            // stream in
            // vhashing::HashTable<int3, VoxelBlock, BlockHasher, BlockEqual, vhashing::std_memspace>
            // bmhi(*dev_blockmap_chunks);
            // int count = 0;
            // for(int i = 0; i < *(bmhi.heap_counter); i ++){
            //     for(int j = 0; j < 4; j ++){
            //         int offset = i * 4 + j;
            //         vhashing::HashEntryBase<int3> &entr = bmhi.hash_table[offset];
            //         int3 ii = entr.key;
            //         if(ii.x == 999999 && ii.y == 999999 && ii.z == 999999 )
            //             continue;
            //         else{
            //             count ++;
            //         }
            //     }
            // }
            // printf("* get heap_counter %d \n", *(bmhi.heap_counter));

            // int recount = 0;
            // // stream in
            // vhashing::HashTable<int3, VoxelBlock, BlockHasher, BlockEqual, vhashing::std_memspace>
            // bmhhh(100001, 4, 400000, int3{999999, 999999, 999999});

            // // check
            // for (int i=0; i<*(bmhi.heap_counter); i++) {
            //     int3 key = bmhi.key_heap[i];
            //     // printf("key\t%d\t%d\t%d\n", key.x, key.y, key.z);
            //     if(bmhhh.find(key) == bmhhh.end()){
            //         VoxelBlock vb;
            //         bmhhh[key] = vb;
            //         recount ++;
            //     }
            // }

            // printf("count out ==== %d\n", recount);
            // for(int i = 0; i < h_inChunkCounter; i ++){
            //     VoxelBlock vb = bmhi[h_inBlockPos[i].pos];
            //     printf("pos = %d %d %d \n", h_inBlockPos[i].pos.x, h_inBlockPos[i].pos.y, h_inBlockPos[i].pos.z);
            // }
        }
    }

    __global__ void getMapValueKernel(vhashing::HashTableBase<int3, VoxelBlock, BlockHasher, BlockEqual> dev_blockmap_,
        VoxelBlock* d_outBlock, VoxelBlockPos* d_outBlockPosHeap){
        const unsigned int idheap = blockIdx.x * blockDim.x + threadIdx.x;
        if(idheap < *(dev_blockmap_.heap_counter)){
            int3 pos = dev_blockmap_.key_heap[idheap];
            d_outBlockPosHeap[idheap].pos = pos;
            d_outBlock[idheap] = dev_blockmap_[pos];
        }
    }

    __host__ void GpuTsdfGenerator::streamOutGPU2CPU(){

        VoxelBlock* d_outBlock;
        VoxelBlockPos* d_outBlockPosHeap;


        cudaSafeCall(cudaMalloc(&d_outBlock, sizeof(VoxelBlock) * h_heapBlockCounter));
        cudaSafeCall(cudaMalloc(&d_outBlockPosHeap, sizeof(VoxelBlockPos) * h_heapBlockCounter));
        
        {

            const dim3 grid_size((h_heapBlockCounter + T_PER_BLOCK - 1) / T_PER_BLOCK, 1);
            const dim3 block_size(T_PER_BLOCK, 1);

            getMapValueKernel <<< grid_size, block_size >>> (*dev_blockmap_, d_outBlock, d_outBlockPosHeap);

        }

        VoxelBlock* h_outBlock;
        VoxelBlockPos* h_outBlockPosHeap;

        h_outBlock = new VoxelBlock[h_heapBlockCounter];
        h_outBlockPosHeap = new VoxelBlockPos[h_heapBlockCounter];

        cudaSafeCall(cudaMemcpy(h_outBlock, d_outBlock, sizeof(VoxelBlock) * h_heapBlockCounter, cudaMemcpyDeviceToHost));
        cudaSafeCall(cudaMemcpy(h_outBlockPosHeap, d_outBlockPosHeap, sizeof(VoxelBlockPos) * h_heapBlockCounter, cudaMemcpyDeviceToHost));

        cudaFree(d_outBlock);
        cudaFree(d_outBlockPosHeap);

        std::unique_lock<std::mutex> lock(tri_mutex_);

        // vhashing::HashTable<int3, VoxelBlock, BlockHasher, BlockEqual, vhashing::std_memspace>
        // h_blockmap_(*dev_blockmap_);


        // std::unique_lock<std::mutex> lock(tri_mutex_);
        // std::unique_lock<std::mutex> lock(tsdf_mutex_);

        int counter = h_heapBlockCounter;

        int total_vox = VOXEL_PER_BLOCK * VOXEL_PER_BLOCK * VOXEL_PER_BLOCK;

        for(int i = 0; i < counter; i ++){
            int3 pos = h_outBlockPosHeap[i].pos;
            VoxelBlock& vb = h_outBlock[i];
            int3 startpos = pos * make_int3(VOXEL_PER_BLOCK);
            int3 chunkpos = block2chunk(pos);
            int idChunk = chunkGetLinearIdx(chunkpos.x, chunkpos.y, chunkpos.z);

            // if(chunkpos.x == 3 && chunkpos.y == 1 && chunkpos.z == -6){
            //     printf("stream out Frame %d:  3 1 -6 in hash\n", countFrame);
            // }
            h_chunks[idChunk].isOccupied = true;

            // printf("block(%d,%d,%d) in chunk %d at (%d,%d,%d)\n", pos.x, pos.y, pos.z, idChunk, chunkpos.x, chunkpos.y, chunkpos.z);
            // assert(h_chunks[idChunk].isOnGPU == 1);

            int idBlock = blockPos2Linear(pos);

            int3 iddblock = blockLinear2Int3(idBlock) + chunkpos * BLOCK_PER_CHUNK;

            // printf("(%d,%d,%d)   and  (%d,%d,%d)\n", iddblock.x, iddblock.y, iddblock.z, pos.x, pos.y, pos.z);
            // assert(iddblock.x == pos.x && iddblock.y == pos.y && iddblock.z == pos.z);

            memcpy(&(h_chunks[idChunk].blocks[idBlock]), &vb, sizeof(Voxel) * total_vox);

            Triangle* tri_src = &hash_tri_[i * total_vox * 5];

            Triangle* tri_dst = &(h_chunks[idChunk].tri_[idBlock * total_vox * 5]);

            memcpy(tri_dst, tri_src, sizeof(Triangle) * total_vox * 5);

            // printf("%f copy to %f\n", h_chunks[idChunk].blocks->voxels[5].sdf, vb.voxels[5].sdf);

            // assert(h_chunks[idChunk].blocks->voxels[0].sdf == vb.voxels[0].sdf);
        }

        free(hash_tri_);

        delete[] h_outBlock;
        delete[] h_outBlockPosHeap;




        // cudaSafeCall(cudaDeviceSynchronize()); //debug

        // std::ofstream outFile;

        // outFile.open("triangles_chunk"+ std::to_string(countFrame) + ".txt");

        // {
        //     int count = 0;
        //     int chunk_half = MAX_CHUNK_NUM / 2;
        //     int block_total = BLOCK_PER_CHUNK * BLOCK_PER_CHUNK * BLOCK_PER_CHUNK;
        //     int total_vox = VOXEL_PER_BLOCK * VOXEL_PER_BLOCK * VOXEL_PER_BLOCK * block_total;
        //     int tri_num = total_vox * 5;
        //     int empty = 0;
        //     for(int x = - chunk_half; x < chunk_half; x ++){
        //         for(int y = - chunk_half; y < chunk_half; y ++){
        //             for(int z = - chunk_half; z < chunk_half; z ++){
        //                 int id = chunkGetLinearIdx(x,y,z);
        //                 Triangle* tri_ = h_chunks[id].tri_;
        //                 if(tri_ != nullptr){
        //                     outFile << "-----------\t"<< x << "\t" <<y<<"\t"<<z<<"\t"<<"-----------\n";

        //                     int count = 0;
        //                     for(int i = 0; i < tri_num; i ++){
        //                         if(!h_chunks[id].tri_[i].valid)
        //                             continue;
        //                         for (int j = 0; j < 3; ++j){
        //                             outFile << tri_[i].p[j].x << "\t" << tri_[i].p[j].y << "\t" << tri_[i].p[j].z << "\t";
        //                         }
        //                         outFile << "\n";
        //                     }
        //                 }

        //             }
        //         }
        //     }
        // }

        // outFile.close();

        // delete h_blockmap_;

    }


    __global__ void IntegrateHashKernel(float *K, float *c2w, float *depth, unsigned char *rgb,
                   int height, int width, MarchingCubeParam *param,  
                   vhashing::HashTableBase<int3, VoxelBlock, BlockHasher, BlockEqual> dev_blockmap_,
                   VoxelBlockPos* d_inBlockPosHeap, unsigned int *d_heapBlockCounter){

        unsigned int idheap = blockIdx.x;

        // if(idheap != 80 || threadIdx.x != 0)
        //     return;
        // printf("heap\t%d\tvoxel\t%d\n",idheap, threadIdx.x);
        if(idheap < *d_heapBlockCounter){
            int3 idBlock = dev_blockmap_.key_heap[idheap];
            VoxelBlock& vb = dev_blockmap_[idBlock];
            Voxel& voxel = vb.voxels[threadIdx.x];

            int VOXEL_PER_BLOCK2 = VOXEL_PER_BLOCK * VOXEL_PER_BLOCK;
            int z = threadIdx.x % VOXEL_PER_BLOCK;
            int y = ((threadIdx.x - z) % VOXEL_PER_BLOCK2) / VOXEL_PER_BLOCK;
            int x = threadIdx.x / VOXEL_PER_BLOCK2;

            // // printf("block\t%d\tthread\t%d\trank\t%d\tvoxel\t(%d,%d,%d)\n", blockIdx.x, threadIdx.x, blockIdx.x*blockDim.x+threadIdx.x, x, y, z);

            int3 idVoxel = idBlock * VOXEL_PER_BLOCK + make_int3(x,y,z);

            float3 voxelpos = voxel2world(idVoxel, param->vox_size);

            // float pt_base_x = param->vox_origin.x + voxelpos.x;
            // float pt_base_y = param->vox_origin.y + voxelpos.y;
            // float pt_base_z = param->vox_origin.z + voxelpos.z;

            // // Convert from base frame camera coordinates to current frame camera coordinates
            // float tmp_pt[3] = {0};
            // tmp_pt[0] = pt_base_x - c2w[0 * 4 + 3];
            // tmp_pt[1] = pt_base_y - c2w[1 * 4 + 3];
            // tmp_pt[2] = pt_base_z - c2w[2 * 4 + 3];
            // float pt_cam_x =
            //         c2w[0 * 4 + 0] * tmp_pt[0] + c2w[1 * 4 + 0] * tmp_pt[1] + c2w[2 * 4 + 0] * tmp_pt[2];
            // float pt_cam_y =
            //         c2w[0 * 4 + 1] * tmp_pt[0] + c2w[1 * 4 + 1] * tmp_pt[1] + c2w[2 * 4 + 1] * tmp_pt[2];
            // float pt_cam_z =
            //         c2w[0 * 4 + 2] * tmp_pt[0] + c2w[1 * 4 + 2] * tmp_pt[1] + c2w[2 * 4 + 2] * tmp_pt[2];

            // int pt_pix_x = roundf(K[0 * 3 + 0] * (pt_cam_x / pt_cam_z) + K[0 * 3 + 2]);
            // int pt_pix_y = roundf(K[1 * 3 + 1] * (pt_cam_y / pt_cam_z) + K[1 * 3 + 2]);
                        // Convert from base frame camera coordinates to current frame camera coordinates
            float pt_base[3] = {voxelpos.x,// + param->vox_origin.x, 
            voxelpos.y,// + param->vox_origin.y, 
            voxelpos.z};// + param->vox_origin.z};// voxel2world(idVoxel, param->vox_size);

            float tmp_pt[3] = {0};
            tmp_pt[0] = pt_base[0] - c2w[0 * 4 + 3];
            tmp_pt[1] = pt_base[1] - c2w[1 * 4 + 3];
            tmp_pt[2] = pt_base[2] - c2w[2 * 4 + 3];
            float pt_cam_x =
                    c2w[0 * 4 + 0] * tmp_pt[0] + c2w[1 * 4 + 0] * tmp_pt[1] + c2w[2 * 4 + 0] * tmp_pt[2];
            float pt_cam_y =
                    c2w[0 * 4 + 1] * tmp_pt[0] + c2w[1 * 4 + 1] * tmp_pt[1] + c2w[2 * 4 + 1] * tmp_pt[2];
            float pt_cam_z =
                    c2w[0 * 4 + 2] * tmp_pt[0] + c2w[1 * 4 + 2] * tmp_pt[1] + c2w[2 * 4 + 2] * tmp_pt[2];
            float pt_cam[3] = {pt_cam_x, pt_cam_y, pt_cam_z};

            base2cam(pt_base, pt_cam, c2w);
            pt_cam_x = pt_cam[0];
            pt_cam_y = pt_cam[1];
            pt_cam_z = pt_cam[2];

            int pt_pix[2];
            cam2frame(pt_cam, pt_pix, K);

            int pt_pix_x = pt_pix[0];
            int pt_pix_y = pt_pix[1];

            // if(pt_pix_x == DEBUG_WIDTH && pt_pix_y == DEBUG_HEIGHT){
            //     float depth_val = depth[pt_pix_y * width + pt_pix_x];
            //     float diff = depth_val - pt_cam_z;
            //     int3 idVoxel = world2voxel(make_float3(pt_base[0], pt_base[1], pt_base[2]), param->vox_size);
            //     printf("idVoxel\t(%d,%d,%d)\tpos\t(%f,%f,%f)\tcamera\t(%f,%f,%f)\tscreen\t(%d,%d)\tdepth\t%f\tdiff\t%f\n",
            //     idVoxel.x, idVoxel.y, idVoxel.z, 
            //     pt_base[0], pt_base[1], pt_base[2], 
            //     pt_cam_x, pt_cam_y, pt_cam_z,
            //     pt_pix_x, pt_pix_y,
            //     depth_val, diff);

            //     if (pt_cam_z <= 0){
            //         printf("1 %d <= 0\n", pt_cam_z);
            //         return;
            //     }


            //     if (pt_pix_x < 0 || pt_pix_x >= width || pt_pix_y < 0 || pt_pix_y >= height){
            //         printf("2  %d %d \n",pt_pix_x,pt_pix_y);
            //         return;
            //     }

            //     if (depth_val <= 0 || depth_val > param->max_depth){
            //         printf("3 %f %f\n", depth_val,  param->max_depth );
            //         return;
            //     }

            //     if (diff <= -param->trunc_margin){
            //         printf("4 %f %f\n", diff,  -param->trunc_margin);
            //         return;
            //     }

            // }


            if (pt_cam_z <= 0)
                return;


            if (pt_pix_x < 0 || pt_pix_x >= width || pt_pix_y < 0 || pt_pix_y >= height)
                return;

            float depth_val = depth[pt_pix_y * width + pt_pix_x];

            if (depth_val <= 0 || depth_val > param->max_depth)
                return;

            float diff = depth_val - pt_cam_z;

            if (diff <= -param->trunc_margin)
                return;

            // if(pt_pix_x % DEBUG_WIDTH == 0 && pt_pix_y % DEBUG_HEIGHT == 0){
            //     float depth_val = depth[pt_pix_y * width + pt_pix_x];
            //     float diff = depth_val - pt_cam_z;
            //     int3 idVoxel = world2voxel(make_float3(pt_base[0], pt_base[1], pt_base[2]), param->vox_size);
            //     printf("good idVoxel\t(%d,%d,%d)\tpos\t(%f,%f,%f)\tcamera\t(%f,%f,%f)\tscreen\t(%d,%d)\tdepth\t%f\tdiff\t%f\n",
            //     idVoxel.x, idVoxel.y, idVoxel.z, 
            //     pt_base[0], pt_base[1], pt_base[2], 
            //     pt_cam_x, pt_cam_y, pt_cam_z,
            //     pt_pix_x, pt_pix_y,
            //     depth_val, diff);
            //     // printf("", diff,  -param->trunc_margin);
            // }

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

            // if(idheap == 100)
            //     printf("sdf\t%f\tcolor\t%d\t%d\t%d\tweight%f\n", voxel.sdf,voxel.sdf_color[0],voxel.sdf_color[1],voxel.sdf_color[2],voxel.weight);

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
            float pt_base[3];
            // pt_base[0] = pt_grid_x * param->vox_size;//param->vox_origin.x + pt_grid_x * param->vox_size;
            // pt_base[1] = pt_grid_y * param->vox_size;//param->vox_origin.y + pt_grid_y * param->vox_size;
            // pt_base[2] = pt_grid_z * param->vox_size;//param->vox_origin.z + pt_grid_z * param->vox_size;
            pt_base[0] = param->vox_origin.x + pt_grid_x * param->vox_size;
            pt_base[1] = param->vox_origin.y + pt_grid_y * param->vox_size;
            pt_base[2] = param->vox_origin.z + pt_grid_z * param->vox_size;

            // Convert from base frame camera coordinates to current frame camera coordinates
            float tmp_pt[3] = {0};
            tmp_pt[0] = pt_base[0] - c2w[0 * 4 + 3];
            tmp_pt[1] = pt_base[1] - c2w[1 * 4 + 3];
            tmp_pt[2] = pt_base[2] - c2w[2 * 4 + 3];
            float pt_cam_x =
                    c2w[0 * 4 + 0] * tmp_pt[0] + c2w[1 * 4 + 0] * tmp_pt[1] + c2w[2 * 4 + 0] * tmp_pt[2];
            float pt_cam_y =
                    c2w[0 * 4 + 1] * tmp_pt[0] + c2w[1 * 4 + 1] * tmp_pt[1] + c2w[2 * 4 + 1] * tmp_pt[2];
            float pt_cam_z =
                    c2w[0 * 4 + 2] * tmp_pt[0] + c2w[1 * 4 + 2] * tmp_pt[1] + c2w[2 * 4 + 2] * tmp_pt[2];
            float pt_cam[3] = {pt_cam_x, pt_cam_y, pt_cam_z};

            base2cam(pt_base, pt_cam, c2w);
            pt_cam_x = pt_cam[0];
            pt_cam_y = pt_cam[1];
            pt_cam_z = pt_cam[2];

            int pt_pix[2];
            cam2frame(pt_cam, pt_pix, K);

            int pt_pix_x = pt_pix[0];//roundf(K[0 * 3 + 0] * (pt_cam_x / pt_cam_z) + K[0 * 3 + 2]);
            int pt_pix_y = pt_pix[1];//roundf(K[1 * 3 + 1] * (pt_cam_y / pt_cam_z) + K[1 * 3 + 2]);

            if(pt_grid_z == 10 && pt_grid_y == 20 && pt_grid_x == 30){
                float cam_pos[3];
                frame2cam(pt_pix, pt_cam_z, cam_pos, K);
                printf("%f %f %f ==== %f %f %f \n", cam_pos[0], cam_pos[1], cam_pos[2], pt_cam[0], pt_cam[1], pt_cam[2]);
                float base_pos[3];
                cam2base(cam_pos, base_pos, c2w);
                printf("%f %f %f ==== %f %f %f \n", base_pos[0], base_pos[1], base_pos[2], pt_base[0], pt_base[1], pt_base[2]);
            }


            if (pt_cam_z <= 0)
                continue;

            // int pt_pix_x = roundf(K[0 * 3 + 0] * (pt_cam_x / pt_cam_z) + K[0 * 3 + 2]);
            // int pt_pix_y = roundf(K[1 * 3 + 1] * (pt_cam_y / pt_cam_z) + K[1 * 3 + 2]);
            if (pt_pix_x < 0 || pt_pix_x >= width || pt_pix_y < 0 || pt_pix_y >= height)
                continue;

            float depth_val = depth[pt_pix_y * width + pt_pix_x];

            if (depth_val <= 0 || depth_val > param->max_depth)
                continue;

            float diff = depth_val - pt_cam_z;

            if (diff <= -param->trunc_margin)
                continue;

            if(pt_pix_x % DEBUG_WIDTH == 0 && pt_pix_y % DEBUG_HEIGHT == 0){
                // int pt_pix_x = roundf(K[0 * 3 + 0] * (pt_cam_x / pt_cam_z) + K[0 * 3 + 2]);
                // int pt_pix_y = roundf(K[1 * 3 + 1] * (pt_cam_y / pt_cam_z) + K[1 * 3 + 2]);
                float depth_val = depth[pt_pix_y * width + pt_pix_x];
                float diff = depth_val - pt_cam_z;
                int3 idx3 = world2voxel(make_float3((float)pt_base[0], (float)pt_base[1], (float)pt_base[2]), param->vox_size);
                int3 idBlock = voxel2block(idx3);
                printf("good pos\t(%f,%f,%f)\tblock\t(%d,%d,%d)\tvoxel\t(%d,%d,%d)\tcamera\t(%f,%f,%f)\tscreen\t(%d,%d)\tdepth\t%f\tdiff\t%f\n",
                pt_base[0], pt_base[1], pt_base[2], 
                idBlock.x, idBlock.y, idBlock.z,
                idx3.x, idx3.y, idx3.z,
                pt_cam_x, pt_cam_y, pt_cam_z,
                pt_pix_x, pt_pix_y,
                depth_val, diff);
            }

                // int3 idx3 = world2voxel(make_float3((float)pt_grid_x * param->vox_size, (float)pt_grid_y * param->vox_size, (float)pt_grid_z * param->vox_size), param->vox_size);
                // int3 idBlock = voxel2block(idx3);
                // printf("good pos\t(%f,%f,%f)\tblock\t(%d,%d,%d)\tvoxel\t(%d,%d,%d)\tcamera\t(%f,%f,%f)\tscreen\t(%d,%d)\tdepth\t%f\tdiff\t%f\n",
                // pt_base[0], pt_base[1], pt_base[2], 
                // idBlock.x, idBlock.y, idBlock.z,
                // idx3.x, idx3.y, idx3.z,
                // pt_cam_x, pt_cam_y, pt_cam_z,
                // pt_pix_x, pt_pix_y,
                // depth_val, diff);


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

            // printf("sdf\t%f\tcolor\t%d\t%d\t%d\tweight%f\n", TSDF[volume_idx],TSDF_color[volume_idx * 3],TSDF_color[volume_idx * 3 + 1],TSDF_color[volume_idx * 3 + 2],weight[volume_idx]);

        }
    }
  
        __host__ __device__
    bool operator==(const Vertex &a, const Vertex &b){

        return (a.x == b.x && a.y == b.y && a.z == b.z);
    }


    __device__ int d_floor(float f){
        if(f >= 0.0)
            return (int)f;
        else
            return (int)(f - 0.5);
    }

    __global__ 
    void marchingCubeHashKernel(vhashing::HashTableBase<int3, VoxelBlock, BlockHasher, BlockEqual> dev_blockmap_,
                unsigned int* d_valid_tri, unsigned int *d_heapBlockCounter, Triangle *tri, MarchingCubeParam *param){

        unsigned int idheap = blockIdx.x;

        // if(idheap != 100 || threadIdx.x != 7)
        //     return;
        // printf("heap\t%d\tvoxel\t%d\n",idheap, threadIdx.x);
        if(idheap < *d_heapBlockCounter){   
            // if(blockIdx.x == 0 && threadIdx.x == 0)
            //     printf("id heap == %d\n", idheap);

            int3 idBlock = dev_blockmap_.key_heap[idheap];
            VoxelBlock& vb = dev_blockmap_[idBlock];
            // Voxel& voxel = vb.voxels[threadIdx.x];
            // if(blockIdx.x == 0 && threadIdx.x == 0)
            //     printf("vb  ==\n");

            int VOXEL_PER_BLOCK2 = VOXEL_PER_BLOCK * VOXEL_PER_BLOCK;
            int z = threadIdx.x % VOXEL_PER_BLOCK + idBlock.z * VOXEL_PER_BLOCK;
            int y = ((threadIdx.x - z) % VOXEL_PER_BLOCK2) / VOXEL_PER_BLOCK + idBlock.y * VOXEL_PER_BLOCK;
            int x = threadIdx.x / VOXEL_PER_BLOCK2 + idBlock.x * VOXEL_PER_BLOCK;

            GRIDCELL grid;

            // printf("%d %d start \n",blockIdx.x, threadIdx.x);

            for (int k = 0; k < 8; ++k) {

                int cxi = x + param->idxMap[k][0];
                int cyi = y + param->idxMap[k][1];
                int czi = z + param->idxMap[k][2];

                int3 id_nb_block = voxel2block(make_int3(cxi,cyi,czi));

                // printf("current block\t%d\t%d\t%d\tvoxel\t%d\t%d\t%d\tnbvoxel\t%d\t%d\t%d\tblock\t%d\t%d\t%d\n", idBlock.x, idBlock.y, idBlock.z, x, y, z, cxi, cyi, czi, id_nb_block.x, id_nb_block.y, id_nb_block.z);
                if(cxi == x)
                    assert(idBlock.x == id_nb_block.x);
                if(cyi == y)
                    assert(idBlock.y == id_nb_block.y);                
                if(czi == z)
                    assert(idBlock.z == id_nb_block.z);

                int linear_id = voxelLinearInBlock(make_int3(cxi, cyi,czi), id_nb_block);

                if(dev_blockmap_.find(id_nb_block) != dev_blockmap_.end()){
                    grid.p[k] = Vertex(cxi, cyi, czi);

                    // if(blockIdx.x == 0 && threadIdx.x == 0)
                    //     printf(" hhhh \n");

                    VoxelBlock& nb_block = dev_blockmap_[id_nb_block];


                    // if(blockIdx.x == 0 && threadIdx.x == 0)
                    //     printf(" hhh00h \n");

                    Voxel& nb_voxel = nb_block.voxels[linear_id];


                    // if(blockIdx.x == 0 && threadIdx.x == 0)
                    //     printf(" hhxxhh \n");

                    grid.p[k].r = nb_voxel.sdf_color[0];
                    grid.p[k].g = nb_voxel.sdf_color[1];
                    grid.p[k].b = nb_voxel.sdf_color[2];
                    grid.val[k] = nb_voxel.sdf;

                    // if(x == DEBUG_X && y == DEBUG_Y && z == DEBUG_Z)
                    // if(blockIdx.x == 0 && threadIdx.x == 0)
                    //     printf("nbvoxel\t%d\t%d\t%d\tr%dg%db%d\tsdf%f\n", cxi, cyi, czi,grid.p[k].r,grid.p[k].g,grid.p[k].b, grid.val[k]);
                    //     // printf("%d\tnbvoxel\t%d\t%d\t%d\tsdf=%f\n", linear_id, cxi, cyi, czi,voxel.sdf);
                }else{
                    // grid.p[k].r = 0;
                    // grid.p[k].g = 0;
                    // grid.p[k].b = 0;
                    // grid.val[k] = 0;

                    // if(x == DEBUG_X && y == DEBUG_Y && z == DEBUG_Z)
                    // if(x % 4 == 0 && y % 4 == 0 && z % 4 == 0)
                    // if(blockIdx.x == 0 && threadIdx.x == 0)
                    //     printf("return here\n");
                    // printf("%d %d return \n",blockIdx.x, threadIdx.x);

                    return;
                        // printf("%d\tn_____l\t%d\t%d\t%d\tsdf=%f\n", linear_id, cxi, cyi, czi, grid.val[k]);
                }

                // if(pt_grid_x == 20 && pt_grid_y == 8 && pt_grid_z == 0){
                //     printf("80 34 3  tsdf %f \n", grid.val[k]);
                // }
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
                return;

            // if(blockIdx.x == 0 && threadIdx.x == 0)
            //     printf("valid  ==\n");


            // if(x == 95 && y == 63 && z == 21)
            //     printf("\t%d\t%d\t%d\n", x, y, z);

            // printf("valid sdf\n");

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


            int index = idheap * (VOXEL_PER_BLOCK * VOXEL_PER_BLOCK * VOXEL_PER_BLOCK) + threadIdx.x;

            int count = 0;
            // int addr = 0;
            for (int ti = 0; param->triTable[cubeIndex][ti] != -1; ti += 3) {
                // addr = atomicAdd(&d_valid_tri[0], 1);
                // tri[addr].p[0] = vertlist[param->triTable[cubeIndex][ti]];
                // tri[addr].p[1] = vertlist[param->triTable[cubeIndex][ti + 1]];
                // tri[addr].p[2] = vertlist[param->triTable[cubeIndex][ti + 2]];
                // tri[addr].valid = true;
                tri[index * 5 + count].p[0] = vertlist[param->triTable[cubeIndex][ti]];
                tri[index * 5 + count].p[1] = vertlist[param->triTable[cubeIndex][ti + 1]];
                tri[index * 5 + count].p[2] = vertlist[param->triTable[cubeIndex][ti + 2]];
                tri[index * 5 + count].valid = true;

                if(tri[index * 5 + count].p[0] == tri[index * 5 + count].p[1] || 
                    tri[index * 5 + count].p[1] == tri[index * 5 + count].p[2] || 
                    tri[index * 5 + count].p[0] == tri[index * 5 + count].p[1])
                    tri[index * 5 + count].valid = false;

                count++;
            }

            // if(blockIdx.x == 0 && threadIdx.x == 0)
            // printf("%d %d count  ==  %d   valid == %d\n",blockIdx.x, threadIdx.x, count, addr);


            // if(x % 4 == 0 && y % 4 == 0 && z % 4 == 0){

            //     for (int k = 0; k < 8; ++k) {
            //         int cxi = x + param->idxMap[k][0];
            //         int cyi = y + param->idxMap[k][1];
            //         int czi = z + param->idxMap[k][2];

            //         int3 id_nb_block = voxel2block(make_int3(cxi,cyi,czi));

            //         // printf("current block\t%d\t%d\t%d\tvoxel\t%d\t%d\t%d\tnbvoxel\t%d\t%d\t%d\tblock\t%d\t%d\t%d\n", idBlock.x, idBlock.y, idBlock.z, x, y, z, cxi, cyi, czi, id_nb_block.x, id_nb_block.y, id_nb_block.z);
            //         if(cxi == x)
            //             assert(idBlock.x == id_nb_block.x);
            //         if(cyi == y)
            //             assert(idBlock.y == id_nb_block.y);                
            //         if(czi == z)
            //             assert(idBlock.z == id_nb_block.z);

            //         int linear_id = voxelLinearInBlock(make_int3(cxi, cyi,czi), id_nb_block);

            //         if(dev_blockmap_.find(id_nb_block) != dev_blockmap_.end()){
            //             grid.p[k] = Vertex(cxi, cyi, czi);


            //             VoxelBlock& nb_block = dev_blockmap_[id_nb_block];
            //             Voxel& nb_voxel = nb_block.voxels[linear_id];

            //             grid.p[k].r = nb_voxel.sdf_color[0];
            //             grid.p[k].g = nb_voxel.sdf_color[1];
            //             grid.p[k].b = nb_voxel.sdf_color[2];
            //             grid.val[k] = nb_voxel.sdf;
            //             printf("hashmesh\t%d\t%d\t%d\tr%dg%db%d\tsdf%f\n", cxi, cyi, czi,grid.p[k].r,grid.p[k].g,grid.p[k].b, grid.val[k]);

            //         }
            //     }

            // }

            // if(x == DEBUG_X && y == DEBUG_Y && z == DEBUG_Z)
            //     printf("hashmesh\t%d\t%d\t%d\tcount=%d\n", x, y, z, count);

            // printf("count tri %d\n", index * 5 + count);
            // assert(count != 0);
        }
    }

    __global__
    void marchingCubeKernel(float *TSDF, unsigned char *TSDF_color, Triangle *tri, MarchingCubeParam *param) {
        int pt_grid_z = blockIdx.x;
        int pt_grid_y = threadIdx.x;

        int global_index = pt_grid_z * param->vox_dim.x * param->vox_dim.y - pt_grid_y * param->vox_dim.x;

        for (int pt_grid_x = 0; pt_grid_x < param->vox_dim.x; ++pt_grid_x) {
            int index = global_index + pt_grid_x;

            // if(pt_grid_x != 112 || pt_grid_y != 33 || pt_grid_z != 1)
            //     return;

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

                if(pt_grid_x == DEBUG_X && pt_grid_y == DEBUG_Y && pt_grid_z == DEBUG_Z)
                    printf("nbvoxel\t%d\t%d\t%d\tr%dg%db%d\tsdf%f\n", cxi, cyi, czi,grid.p[k].r,grid.p[k].g,grid.p[k].b, grid.val[k]);

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

            // if(pt_grid_x == DEBUG_X && pt_grid_y == DEBUG_Y && pt_grid_z == DEBUG_Z)
                // printf("mesh\t%d\t%d\t%d\tcount=%d\n", pt_grid_x, pt_grid_y, pt_grid_z, count);

            if(pt_grid_x % 4 == 0 && pt_grid_y % 4 == 0 && pt_grid_z % 4 == 0){

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

                // if(pt_grid_x == DEBUG_X && pt_grid_y == DEBUG_Y && pt_grid_z == DEBUG_Z)
                    // printf("mesh\t%d\t%d\t%d\tr%dg%db%d\tsdf%f\n", cxi, cyi, czi,grid.p[k].r,grid.p[k].g,grid.p[k].b, grid.val[k]);

            }
        }

            assert(count != 0);
        }
    }

    __host__
    void GpuTsdfGenerator::clearheap(){
        // unsigned int src = 0;
        // // cudaSafeCall(cudaFree(d_heapBlockCounter));
        // cudaSafeCall(cudaMemcpy(d_heapBlockCounter, &src, sizeof(unsigned int), cudaMemcpyHostToDevice));
        // std::cout<<" d_heapBlockCounter clear "<<std::endl;
        dev_blockmap_->clearheap();
        dev_blockmap_chunks->clearheap();
        // d_heapBlockCounter = 0;
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

        // cudaSafeCall(cudaMalloc(&d_outBlock, sizeof(VoxelBlock) * MAX_CPU2GPU_BLOCKS));
        // std::cout<<" d_outBlock init "<<std::endl;
        // cudaSafeCall(cudaMalloc(&d_outBlockPos, sizeof(VoxelBlockPos) * MAX_CPU2GPU_BLOCKS));
        // std::cout<<" d_outBlockPos init "<<std::endl;
        cudaSafeCall(cudaMalloc(&d_inBlock, sizeof(VoxelBlock) * MAX_CPU2GPU_BLOCKS));
        std::cout<<" d_inBlock init "<<std::endl;
        cudaSafeCall(cudaMalloc(&d_inBlockPos, sizeof(VoxelBlockPos) * MAX_CPU2GPU_BLOCKS));
        std::cout<<" d_inBlockPos init "<<std::endl;
        cudaSafeCall(cudaMalloc(&d_inBlockPosHeap, sizeof(VoxelBlockPos) * MAX_CPU2GPU_BLOCKS));
        std::cout<<" d_inBlockPosHeap init "<<std::endl;
        // cudaSafeCall(cudaMalloc(&d_outBlockCounter, sizeof(unsigned int)));
        // std::cout<<" d_outBlockCounter init "<<std::endl;
        d_heapBlockCounter = NULL;
        cudaSafeCall(cudaMalloc((void**)&d_heapBlockCounter, sizeof(unsigned int)));

        // vhashing::HashTable<int3, VoxelBlock, BlockHasher, BlockEqual, vhashing::device_memspace>
        //     blocks(10000, 2, 19997, int3{999999, 999999, 999999});

        // dev_blockmap_ = new vhashing::HashTable<int3, VoxelBlock, BlockHasher, BlockEqual, vhashing::device_memspace>(100001, 4, 400000, int3{999999, 999999, 999999});
        // std::cout<<" dev_blockmap_ init "<<std::endl;

        // std::cout<<" d_heapBlockCounter init "<<std::endl;
        // dev_blockmap_chunks = new vhashing::HashTable<int3, VoxelBlock, BlockHasher, BlockEqual, vhashing::device_memspace>(100001, 4, 400000, int3{999999, 999999, 999999});
        // std::cout<<" dev_blockmap_chunks init "<<std::endl;

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

        // // Initialize voxel grid
        // TSDF_ = new float[param_->total_vox];
        // TSDF_color_ = new unsigned char[3 * param_->total_vox];
        // weight_ = new float[param_->total_vox];
        // memset(TSDF_, 1.0f, sizeof(float) * param_->total_vox);
        // memset(TSDF_color_, 0.0f, 3 * sizeof(unsigned char) * param_->total_vox);
        // memset(weight_, 0.0f, sizeof(float) * param_->total_vox);

        tri_ = (Triangle *) malloc(sizeof(Triangle) * param_->total_vox * 5);
        
        // Load variables to GPU memory
        cudaMalloc(&dev_param_, sizeof(MarchingCubeParam));
        cudaMalloc(&dev_TSDF_, param_->total_vox * sizeof(float));
        cudaMalloc(&dev_TSDF_color_,
                   3 * param_->total_vox * sizeof(unsigned char));
        cudaMalloc(&dev_weight_,
                   param_->total_vox * sizeof(float));
        // cudaMalloc(&dev_tri_, sizeof(Triangle) * param_->total_vox * 5);
        checkCUDA(__LINE__, cudaGetLastError());

        cudaMemcpy(dev_param_, param_, sizeof(MarchingCubeParam), cudaMemcpyHostToDevice);
        // cudaMemcpy(dev_TSDF_, TSDF_,
                   // param_->total_vox * sizeof(float), cudaMemcpyHostToDevice);
        // cudaMemcpy(dev_TSDF_color_, TSDF_color_,
        //            3 * param_->total_vox * sizeof(unsigned char),
        //            cudaMemcpyHostToDevice);
        // cudaMemcpy(dev_weight_, weight_,
        //            param_->total_vox * sizeof(float), cudaMemcpyHostToDevice);
        checkCUDA(__LINE__, cudaGetLastError());

        cudaMalloc(&dev_K_, 3 * 3 * sizeof(float));
        cudaMemcpy(dev_K_, K_, 3 * 3 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMalloc(&dev_c2w_, 4 * 4 * sizeof(float));
        cudaMalloc(&dev_depth_, im_height_ * im_width_ * sizeof(float));
        cudaMalloc(&dev_rgb_, 3 * im_height_ * im_width_ * sizeof(unsigned char));
        checkCUDA(__LINE__, cudaGetLastError());

    }

    // __host__
    // void inverse_matrix(float* m, float* minv){
    //     // computes the inverse of a matrix m
    //     std::cout<<" m = "<<std::endl;

    //     for(int r=0;r<3;++r){
    //         for(int c=0;c<3;++c){
    //             // m[r * 3 + c] = c2w_[r * 4 + c];
    //             printf("%f\t",m[r * 3 + c]);
    //         }
    //         std::cout<<std::endl;
    //     }

    //     double det = m[0 * 3 + 0] * (m[1 * 3 + 1] * m[2 * 3 + 2] - m[2 * 3 + 1] * m[1 * 3 + 2]) -
    //                  m[0 * 3 + 1] * (m[1 * 3 + 0] * m[2 * 3 + 2] - m[1 * 3 + 2] * m[2 * 3 + 0]) +
    //                  m[0 * 3 + 2] * (m[1 * 3 + 0] * m[2 * 3 + 1] - m[1 * 3 + 1] * m[2 * 3 + 0]);

    //     double invdet = 1 / det;

    //     std::cout<<" inverse "<<std::endl;

    //     // inverse_matrix(m, minv);
    //     for(int r=0;r<3;++r){
    //         for(int c=0;c<3;++c){
    //             printf("%f\t",minv[r * 3 + c]);
    //         }
    //         std::cout<<std::endl;
    //     }

    //     minv[0 * 3 + 0] = (m[1 * 3 + 1] * m[2 * 3 + 2] - m[2 * 3 + 1] * m[1 * 3 + 2]) * invdet;
    //     minv[0 * 3 + 1] = (m[0 * 3 + 2] * m[2 * 3 + 1] - m[0 * 3 + 1] * m[2 * 3 + 2]) * invdet;
    //     minv[0 * 3 + 2] = (m[0 * 3 + 1] * m[1 * 3 + 2] - m[0 * 3 + 2] * m[1 * 3 + 1]) * invdet;
    //     minv[1 * 3 + 0] = (m[1 * 3 + 2] * m[2 * 3 + 0] - m[1 * 3 + 0] * m[2 * 3 + 2]) * invdet;
    //     minv[1 * 3 + 1] = (m[0 * 3 + 0] * m[2 * 3 + 2] - m[0 * 3 + 2] * m[2 * 3 + 0]) * invdet;
    //     minv[1 * 3 + 2] = (m[1 * 3 + 0] * m[0 * 3 + 2] - m[0 * 3 + 0] * m[1 * 3 + 2]) * invdet;
    //     minv[2 * 3 + 0] = (m[1 * 3 + 0] * m[2 * 3 + 1] - m[2 * 3 + 0] * m[1 * 3 + 1]) * invdet;
    //     minv[2 * 3 + 1] = (m[2 * 3 + 0] * m[0 * 3 + 1] - m[0 * 3 + 0] * m[2 * 3 + 1]) * invdet;
    //     minv[2 * 3 + 2] = (m[0 * 3 + 0] * m[1 * 3 + 1] - m[1 * 3 + 0] * m[0 * 3 + 1]) * invdet;
    // }


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


        dev_blockmap_ = new vhashing::HashTable<int3, VoxelBlock, BlockHasher, BlockEqual, vhashing::device_memspace>(100001, 4, 400000, int3{999999, 999999, 999999});
        // std::cout<<" dev_blockmap_ init "<<std::endl;

        // std::cout<<" d_heapBlockCounter init "<<std::endl;
        dev_blockmap_chunks = new vhashing::HashTable<int3, VoxelBlock, BlockHasher, BlockEqual, vhashing::device_memspace>(100001, 4, 400000, int3{999999, 999999, 999999});
        // std::cout<<" dev_blockmap_chunks init "<<std::endl;

        cudaMemcpy(dev_c2w_, c2w, 4 * 4 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_depth_, depth, im_height_ * im_width_ * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_rgb_, rgb, 3 * im_height_ * im_width_ * sizeof(unsigned char), cudaMemcpyHostToDevice);
        checkCUDA(__LINE__, cudaGetLastError());

        memcpy(c2w_, c2w, 4 * 4 * sizeof(float));

        // std::cout<< "clear h_heapBlockCounter" <<std::endl;
        clearheap();

        streamInCPU2GPU(dev_K_, dev_c2w_, dev_depth_);
        checkCUDA(__LINE__, cudaGetLastError());

        HashAssign(dev_depth_, im_height_, im_width_, dev_param_, dev_K_, dev_c2w_);
        // HashAlloc();
        checkCUDA(__LINE__, cudaGetLastError());

        Triangle *dev_hash_tri_ = nullptr;
        unsigned int *d_valid_tri;
        unsigned int h_valid_tri = 0;
        cudaSafeCall(cudaMalloc(&d_valid_tri, sizeof(unsigned int)));
        int tri_0;
        cudaSafeCall(cudaMemcpy(d_valid_tri, &tri_0, sizeof(unsigned int), cudaMemcpyHostToDevice));

        cudaSafeCall(cudaMemcpy(&h_heapBlockCounter,d_heapBlockCounter, sizeof(unsigned int), cudaMemcpyDeviceToHost));
        // std::cout<<"h_heapBlockCounter = "<<h_heapBlockCounter<<std::endl;

        unsigned int total_vox = h_heapBlockCounter * VOXEL_PER_BLOCK * VOXEL_PER_BLOCK * VOXEL_PER_BLOCK;

        {
            std::unique_lock<std::mutex> lock(tsdf_mutex_);
            const dim3 gridSize(h_heapBlockCounter, 1);
            const dim3 blockSize(VOXEL_PER_BLOCK * VOXEL_PER_BLOCK * VOXEL_PER_BLOCK, 1);
            checkCUDA(__LINE__, cudaGetLastError());

            // std::cout<<"start IntegrateHashKernel"<<std::endl;

            IntegrateHashKernel <<< gridSize, blockSize >>> (dev_K_, dev_c2w_, dev_depth_, dev_rgb_, 
                im_height_, im_width_, dev_param_, *dev_blockmap_, d_inBlockPosHeap, d_heapBlockCounter);

            checkCUDA(__LINE__, cudaGetLastError());
            // std::cout<<"finish IntegrateHashKernel"<<std::endl;


            cudaMalloc(&dev_hash_tri_, sizeof(Triangle) * total_vox * 5);
            cudaMemset(dev_hash_tri_, 0, sizeof(Triangle) * total_vox * 5);

            // std::cout<<"start marchingCubeHashKernel"<<std::endl;

            marchingCubeHashKernel << < gridSize, blockSize >> >
                                                       (*dev_blockmap_, d_valid_tri, d_heapBlockCounter, dev_hash_tri_, dev_param_);
            checkCUDA(__LINE__, cudaGetLastError());

            cudaSafeCall(cudaMemcpy(&h_valid_tri, d_valid_tri, sizeof(unsigned int), cudaMemcpyDeviceToHost));

            // std::cout<<"finish marchingCubeHashKernel"<<std::endl;
        }

        
        hash_tri_ = (Triangle *) malloc(sizeof(Triangle) * total_vox * 5);


        // cudaSafeCall(cudaMemcpy(h_inBlockPosHeap, dev_blockmap_->key_heap, sizeof(VoxelBlockPos) * h_heapBlockCounter, cudaMemcpyDeviceToHost));

        {
            std::unique_lock<std::mutex> lock(tri_mutex_);
            // cudaMemcpy(tri_, dev_hash_tri_, sizeof(Triangle) * total_vox * 5, cudaMemcpyDeviceToHost);
            // checkCUDA(__LINE__, cudaGetLastError());
            cudaMemcpy(hash_tri_, dev_hash_tri_, sizeof(Triangle) * total_vox * 5, cudaMemcpyDeviceToHost);
            checkCUDA(__LINE__, cudaGetLastError());
        }

        // std::ofstream outFile;
        // outFile.open("triangles_hash"+ std::to_string(countFrame) + ".txt");

        // int count_tri = 0;
        // for(int i = 0; i < total_vox * 5; i ++){
        //     if(hash_tri_[i].valid){
        //         for (int j = 0; j < 3; ++j){
        //             outFile << hash_tri_[i].p[j].x << "\t" << hash_tri_[i].p[j].y << "\t" << hash_tri_[i].p[j].z << "\t";
        //         }
        //         outFile << "\n";
        //         count_tri ++;
        //     }
        // }
        // outFile.close();

        // std::cout<<"  count_tri == " <<count_tri<<"  total =="<< total_vox * 5<<std::endl;

        // printf("start streamOut\n");
        streamOutGPU2CPU();
        // printf("end streamOut\n");

        cudaFree(dev_hash_tri_);
        cudaFree(d_valid_tri);

        delete dev_blockmap_;
        delete dev_blockmap_chunks;
        // printf(" release space \n");
        //streamOut

        countFrame ++;

    }

    __host__ 
    void GpuTsdfGenerator::insert_tri(){

        // for(int i = 0; i < h_heapBlockCounter; i ++){
        //     for(int k = 0; k < 8; k ++){

        //     }
        // }

        // {
        //     std::unique_lock<std::mutex> lock(tsdf_mutex_);
        //     cudaMemcpy(TSDF_, dev_TSDF_,
        //                param_->total_vox * sizeof(float), cudaMemcpyDeviceToHost);
        //     cudaMemcpy(TSDF_color_, dev_TSDF_color_,
        //                3 * param_->total_vox * sizeof(unsigned char), cudaMemcpyDeviceToHost);

        //     checkCUDA(__LINE__, cudaGetLastError());
        // }
    }

    __host__
    void GpuTsdfGenerator::Shutdown() {
        free(tri_);
        // cudaFree(dev_TSDF_);
        // cudaFree(dev_TSDF_color_);
        // cudaFree(dev_weight_);
        cudaFree(dev_K_);
        cudaFree(dev_c2w_);
        cudaFree(dev_depth_);
        cudaFree(dev_rgb_);
        // cudaFree(dev_tri_);
        cudaFree(dev_param_);
        cudaFree(d_inBlock);
        cudaFree(d_inBlockPos);
        cudaFree(d_inBlockPosHeap);

        delete dev_blockmap_;
        delete dev_blockmap_chunks;
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
        // {
        //     std::unique_lock<std::mutex> lock(tsdf_mutex_);
        //     cudaMemcpy(TSDF_, dev_TSDF_,
        //                param_->total_vox * sizeof(float), cudaMemcpyDeviceToHost);
        //     cudaMemcpy(TSDF_color_, dev_TSDF_color_,
        //                3 * param_->total_vox * sizeof(unsigned char), cudaMemcpyDeviceToHost);

        //     checkCUDA(__LINE__, cudaGetLastError());
        // }
        tsdf2mesh(filename);
    }

    __host__
    void GpuTsdfGenerator::render() {
        std::unique_lock<std::mutex> lock(chunk_mutex_);
        // for (int i = 0; i < param_->total_vox * 5; ++i) {
        //     if (!tri_[i].valid)
        //         continue;
        //     glBegin(GL_TRIANGLES);
        //     for (int j = 0; j < 3; ++j) {
        //         glColor3f(tri_[i].p[j].r / 255.f, tri_[i].p[j].g / 255.f, tri_[i].p[j].b / 255.f);
        //         glVertex3f(10 * tri_[i].p[j].x * param_->vox_size - 25,
        //                    10 * tri_[i].p[j].y * param_->vox_size - 25,
        //                    10 * tri_[i].p[j].z * param_->vox_size - 20);
        //         // printf("\t%d\t%d\t%\n", tri_[i].p[j].x, tri_[i].p[j].y, tri_[i].p[j].z);
        //     }
        //     glEnd();
        // }
        int chunk_half = MAX_CHUNK_NUM / 2;
        int block_total = BLOCK_PER_CHUNK * BLOCK_PER_CHUNK * BLOCK_PER_CHUNK;
        int total_vox = VOXEL_PER_BLOCK * VOXEL_PER_BLOCK * VOXEL_PER_BLOCK * block_total;
        int tri_num = total_vox * 5;
        // int empty = 0;
        for(int x = - chunk_half; x < chunk_half; x ++){
            for(int y = - chunk_half; y < chunk_half; y ++){
                for(int z = - chunk_half; z < chunk_half; z ++){
                    int id = chunkGetLinearIdx(x,y,z);
                    Triangle* tri_ = h_chunks[id].tri_;
                    if(tri_ != nullptr){
                        int count = 0;
                        for(int i = 0; i < tri_num; i ++){
                            if(!h_chunks[id].tri_[i].valid)
                                continue;
                            count ++;
                            glBegin(GL_TRIANGLES);
                            for (int j = 0; j < 3; ++j) {
                                glColor3f(tri_[i].p[j].r / 255.f, tri_[i].p[j].g / 255.f, tri_[i].p[j].b / 255.f);
                                glVertex3f(10 * tri_[i].p[j].x * param_->vox_size - 25,
                                           10 * tri_[i].p[j].y * param_->vox_size - 25,
                                           10 * tri_[i].p[j].z * param_->vox_size - 20);
                            }
                            glEnd();
                        }
                        if(count == 0)
                            h_chunks[id].isOccupied = false;
                    }

                }
            }
        }
    }

    __host__
    void GpuTsdfGenerator::tsdf2mesh(std::string outputFileName) {
        std::vector<Face> faces;
        std::vector<Vertex> vertices;

        // std::unordered_map<std::string, int> verticesIdx;
        // std::vector<std::list<std::pair<Vertex, int>>> hash_table(param_->total_vox,
                                                                  // std::list<std::pair<Vertex, int>>());
        std::unique_lock<std::mutex> lock(chunk_mutex_);

        // int vertexCount = 0;
        // int emptyCount = 0;
        int totalsize = 0;
        int validCount = 0;

        std::cout << "Start saving ply, totalsize: " << param_->total_vox << std::endl;

        std::unordered_map<Vertex, int, VertexHasher, VertexEqual> vertexHashTable;

        int chunk_half = MAX_CHUNK_NUM / 2;
        int block_total = BLOCK_PER_CHUNK * BLOCK_PER_CHUNK * BLOCK_PER_CHUNK;
        int total_vox = VOXEL_PER_BLOCK * VOXEL_PER_BLOCK * VOXEL_PER_BLOCK * block_total;
        // int tri_num = total_vox * 5;

        int VoxelPerChunk = BLOCK_PER_CHUNK * VOXEL_PER_BLOCK * BLOCK_PER_CHUNK * VOXEL_PER_BLOCK * BLOCK_PER_CHUNK * VOXEL_PER_BLOCK;

        for(int x = - chunk_half; x < chunk_half; x ++){
            for(int y = - chunk_half; y < chunk_half; y ++){
                for(int z = - chunk_half; z < chunk_half; z ++){
                    int id = chunkGetLinearIdx(x,y,z);
                    if(h_chunks[id].isOccupied == false)
                        h_chunks[id].release();

                    // printf("id chunk %d \n", id);
                    Triangle* tri_ = h_chunks[id].tri_;
                    if(tri_ != nullptr){
                        totalsize += VoxelPerChunk;


                        // int count = 0;

                        for(int k = 0; k < total_vox; k ++){
                            int flag = 0;
                            for(int i = 0; i < 5; i ++){
                                int pi = 5 * k + i;
                                if(!h_chunks[id].tri_[pi].valid)
                                    continue;
                                flag = 1;
                                Face f;
                                for (int j = 0; j < 3; ++j) {
                                    if(vertexHashTable.find(tri_[pi].p[j]) == vertexHashTable.end()){
                                        unsigned int count = vertices.size();
                                        vertexHashTable[tri_[pi].p[j]] = count;
                                        f.vIdx[j] = count;
                                        Vertex vp = tri_[pi].p[j];
                                        vp.x *= param_->vox_size;
                                        vp.y *= param_->vox_size;
                                        vp.z *= param_->vox_size;
                                        vertices.push_back(vp);
                                    }else{
                                        f.vIdx[j] = vertexHashTable[tri_[pi].p[j]];
                                    }
                                }
                                faces.push_back(f);
                            }
                            if(flag)
                                validCount ++;
                        }
                    }

                }
            }
        }

        // for (size_t i = 0; i < param_->total_vox; ++i) {
        //     int zi = i / (param_->vox_dim.x * param_->vox_dim.y);
        //     int yi = (i - zi * param_->vox_dim.x * param_->vox_dim.y) / param_->vox_dim.x;
        //     int xi = i - zi * param_->vox_dim.x * param_->vox_dim.y - yi * param_->vox_dim.x;
        //     if (xi == param_->vox_dim.x - 1 || yi == param_->vox_dim.y - 1 || zi == param_->vox_dim.z - 1)
        //         continue;



        //     /* Create the triangle */
        //     for (int ti = 0; param_->triTable[cubeIndex][ti] != -1; ti += 3) {
        //         Face f;
        //         Triangle t;
        //         t.p[0] = vertlist[param_->triTable[cubeIndex][ti]];
        //         t.p[1] = vertlist[param_->triTable[cubeIndex][ti + 1]];
        //         t.p[2] = vertlist[param_->triTable[cubeIndex][ti + 2]];

        //         uint3 grid_size = make_uint3(param_->vox_dim.x, param_->vox_dim.y, param_->vox_dim.z);
        //         for (int pi = 0; pi < 3; ++pi) {
        //             int idx = find_vertex(t.p[pi], grid_size, param_->vox_size, hash_table);
        //             if (idx == -1) {
        //                 insert_vertex(t.p[pi], vertexCount, grid_size, param_->vox_size, hash_table);
        //                 f.vIdx[pi] = vertexCount++;
        //                 t.p[pi].x = t.p[pi].x * param_->vox_size + param_->vox_origin.x;
        //                 t.p[pi].y = t.p[pi].y * param_->vox_size + param_->vox_origin.y;
        //                 t.p[pi].z = t.p[pi].z * param_->vox_size + param_->vox_origin.z;
        //                 vertices.push_back(t.p[pi]);
        //             } else
        //                 f.vIdx[pi] = idx;
        //         }
        //         faces.push_back(f);
        //     }
        // }


        std::cout << vertices.size() << std::endl;
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
        std::cout << "totalsize = "<< totalsize << " valid = "<< validCount << std::endl;
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

    __device__ float3 wolrd2cam(float* c2w_, float3 pos, MarchingCubeParam* param){
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
        // return true;
        // printf("device print\n");
        float3 pCamera = wolrd2cam(c2w, blocks_pos,param);
        float3 pProj = cameraToKinectProj(pCamera, param);
        //pProj *= 1.5f;    //TODO THIS IS A HACK FIX IT :)
        pProj *= 0.95;
        return !(pProj.x < -1.0f || pProj.x > 1.0f || pProj.y < -1.0f || pProj.y > 1.0f || pProj.z < 0.0f || pProj.z > 1.0f); 
    }

    // __global__ void hashCopyKernel(vhashing::HashTableBase<int3, VoxelBlock, BlockHasher, BlockEqual> dev_blockmap_chunks, 
    //     vhashing::HashTableBase<int3, VoxelBlock, BlockHasher, BlockEqual> dev_blockmap_){
    //     int base = blockDim.x * blockIdx.x  +  threadIdx.x;
    //     vhashing::HashEntryBase<int3> &entr = dev_blockmap_chunks.hash_table[base];
    //     VoxelBlock vb;
    //     dev_blockmap_[entr.key] = vb;//entr.value;
    // }

    __device__ volatile int sem = 0;

    __device__ void acquire_semaphore(volatile int *lock){
      while (atomicCAS((int *)lock, 0, 1) != 0)
        printf("wait\n");
    }

    __device__ void release_semaphore(volatile int *lock){
      // *lock = 0;
      // __threadfence();
        atomicExch((int*)lock, 0);
    }


    __global__ void HashAssignKernel(float *depth, const unsigned int height, const unsigned int width, 
        MarchingCubeParam *param, float* K, float* c2w, 
        vhashing::HashTableBase<int3, VoxelBlock, BlockHasher, BlockEqual> dev_blockmap_chunks, 
        vhashing::HashTableBase<int3, VoxelBlock, BlockHasher, BlockEqual> dev_blockmap_,
        unsigned int *d_heapBlockCounter, VoxelBlockPos* d_inBlockPosHeap){

        const unsigned int x = (blockIdx.x * blockDim.x + threadIdx.x) * DDA_STEP;
        const unsigned int y = (blockIdx.y * blockDim.y + threadIdx.y) * DDA_STEP;



        // if(x != DEBUG_WIDTH || y != DEBUG_HEIGHT)
        //     return;

        // if(x %10 !=0 || y %10 != 0)
        //     return;
    
        // if(dev_blockmap_chunks.find(make_int3(16,16,0)) != dev_blockmap_chunks.end())
        //     printf("insert successfully  16 16 0\n");
        // else
        //     printf("not find 16 16 0\n");
        
        if(x < width && y < height){
            // if(y == 0)
            //     printf("%d\n", x);

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
            raymin = frame2base(x,y,param->min_depth, K, c2w, param);
            raymax = frame2base(x,y,param->max_depth , K, c2w, param);
            float3 camcenter = frame2base(x,y,0, K, c2w, param);
            // printf("raymin\t%f\t%f\t%f\traymax \t%f\t%f\t%f\torigin\t%f\t%f\t%f\n", 
            //         raymin.x, raymin.y, raymin.z, raymax.x, raymax.y, raymax.z, camcenter.x, camcenter.y, camcenter.z);

            float3 rayDir = normalize(raymax - raymin);
            int3 idCurrentBlock = wolrd2block(raymin, param->block_size);
            int3 idEnd = wolrd2block(raymax, param->block_size);

            float3 cam_pos = make_float3(c2w[0 * 4 + 3],c2w[1 * 4 + 3],c2w[2 * 4 + 3]);

            // printf("kernel at(%d,%d), camera(%f,%f,%f), ray(%f,%f,%f) block(%d,%d,%d) to (%d,%d,%d)\n", x, y, cam_pos.x, cam_pos.y, cam_pos.z, rayDir.x, rayDir.y, rayDir.z,
            // idCurrentBlock.x, idCurrentBlock.y, idCurrentBlock.z,
            // idEnd.x, idEnd.y, idEnd.z);

            float3 step = make_float3(sign(rayDir));
            float3 boundarypos = block2world(idCurrentBlock + make_int3(clamp(step, 0.0, 1.0f)), param->block_size) - 0.5f * param->vox_size;
            float3 tmax = (boundarypos - raymin) / rayDir;
            float3 tDelta = (step * param->vox_size * VOXEL_PER_BLOCK) / rayDir;
            int3 idBound = make_int3(make_float3(idEnd) + step);

            if(rayDir.x == 0.0f || boundarypos.x - raymin.x == 0.0f){ tmax.x = PINF; tDelta.x = PINF;}
            if(rayDir.y == 0.0f || boundarypos.y - raymin.y == 0.0f){ tmax.y = PINF; tDelta.y = PINF;}
            if(rayDir.z == 0.0f || boundarypos.z - raymin.z == 0.0f){ tmax.z = PINF; tDelta.z = PINF;}

            unsigned int iter = 0;
            unsigned int maxLoopIterCount = 100;

            while(iter < maxLoopIterCount){
                // if(idCurrentBlock.x == 20 && idCurrentBlock.y == 8 && idCurrentBlock.z == 0)
                //     printf("arrive 20 8 0\n");
                // if(idCurrentBlock.x == 20 && idCurrentBlock.y == 10 && idCurrentBlock.z == 2)
                //     printf("arrive 20 10 2\n");                
                float3 blocks_pos = block2world(idCurrentBlock, param->block_size);
                if(dev_blockmap_chunks.find(idCurrentBlock) != dev_blockmap_chunks.end()){
                    if(isBlockInCameraFrustum(blocks_pos, c2w, param)){

                        //debug
                        int3 chunkpos = block2chunk(idCurrentBlock);
                        // printf("idCurrentBlock (%d,%d,%d) at (%f,%f,%f) in camera\n",idCurrentBlock.x, idCurrentBlock.y, idCurrentBlock.z, blocks_pos.x, blocks_pos.y, blocks_pos.z);
                        // printf("\tin camera");
                        // __syncthreads();
                        // // if (threadIdx.x % 2 == 0)
                        //   acquire_semaphore(&sem);
                        // __syncthreads();
                        // int pre_counter = atomicExch(&d_heapBlockCounter[0], LOCK_HASH);
                        // if(dev_blockmap_.find(idCurrentBlock) == dev_blockmap_.end()){
                        dev_blockmap_[idCurrentBlock] = dev_blockmap_chunks[idCurrentBlock];
                            // uint addr = atomicAdd(&d_heapBlockCounter[0], 1);
                            // d_inBlockPosHeap[addr].pos = idCurrentBlock;
                            // printf("%d %d insert (%d,%d,%d)\n", x, y, idCurrentBlock.x, idCurrentBlock.y, idCurrentBlock.z);
                        // }
                        // __syncthreads();
                        // // if (threadIdx.x % 2 == 0)
                        //   release_semaphore(&sem);
                        // __syncthreads();
                    }
                }

                // // printf("\n");
                // else{
                //     if(isBlockInCameraFrustum(blocks_pos, c2w, param))
                //         printf("not in scope  %d %d %d \n");
                //     else
                //         printf("not in camera frustum\n");
                // }
        
                // if(isBlockInCameraFrustum(blocks_pos, c2w, param)){
                //     // printf("blocks_pos %f %f %f cam_pos %f %f %f in camera ++ %d\n",  blocks_pos.x, blocks_pos.y, blocks_pos.z,
                //     // cam_pos.x, cam_pos.y, cam_pos.z, *d_heapBlockCounter);   ///////bug here
                    
                //     if(dev_blockmap_chunks.find(idCurrentBlock) != dev_blockmap_chunks.end()){
                //         // printf("find block in chunk\n");
                //         uint addr = atomicAdd(&d_heapBlockCounter[0], 1);
                //         dev_blockmap_[idCurrentBlock] = dev_blockmap_chunks[idCurrentBlock];
                //         d_inBlockPosHeap[addr].pos = idCurrentBlock;
                //     }
                //     // else{
                //     //     VoxelBlock vb;
                //     //     dev_blockmap_[idCurrentBlock] = vb;
                //     //     d_inBlockPosHeap[addr].pos = idCurrentBlock;
                //     // }
                // }else{
                //     // printf("blocks_pos %f %f %f cam_pos %f %f %f not in camera\n", blocks_pos.x, blocks_pos.y, blocks_pos.z,
                //     // cam_pos.x, cam_pos.y, cam_pos.z);
                // }

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

    __global__ void getHeapCounterKernel(int* count, 
        vhashing::HashTableBase<int3, VoxelBlock, BlockHasher, BlockEqual> dev_blockmap_){
        if(blockIdx.x == 0 && threadIdx.x == 0){
            *count = *(dev_blockmap_.heap_counter);
            // printf("count kernel 00 == %d \n", *count);
        }
            
    }

    __host__ void GpuTsdfGenerator::HashAssign(float *depth, const unsigned int height, const unsigned int width, 
        MarchingCubeParam *param, float* K, float* c2w){
        
        
        // vhashing::HashTable<int3, VoxelBlock, BlockHasher, BlockEqual, vhashing::std_memspace>
        // bmhi_chunk(*dev_blockmap_chunks);
        

        {
            int dda_step = DDA_STEP;
            const dim3 grid_size((im_width_ / dda_step  + T_PER_BLOCK - 1) / T_PER_BLOCK, (im_height_ / dda_step + T_PER_BLOCK - 1) / T_PER_BLOCK, 1);
            const dim3 block_size(T_PER_BLOCK, T_PER_BLOCK, 1);

            unsigned int dst = 0;
            cudaSafeCall(cudaMemcpy(&dst, d_heapBlockCounter, sizeof(unsigned int), cudaMemcpyDeviceToHost));
            // printf("start HashAssignKernel | d_heapBlockCounter\t%d\n", dst);

            HashAssignKernel <<< grid_size, block_size >>> (dev_depth_, im_height_, im_width_, dev_param_, dev_K_, dev_c2w_, *dev_blockmap_chunks, *dev_blockmap_, d_heapBlockCounter, d_inBlockPosHeap);
            

            // hashCopyKernel <<< 100001, 4 >>> (*dev_blockmap_chunks, *dev_blockmap_);

            cudaSafeCall(cudaMemcpy(&dst, d_heapBlockCounter, sizeof(unsigned int), cudaMemcpyDeviceToHost));
            // printf("Host: HashAssign d_heapBlockCounter %d\n", dst);
        }

        // cudaSafeCall(cudaDeviceSynchronize()); //debug
                    // stream in

       // stream in
        // vhashing::HashTable<int3, VoxelBlock, BlockHasher, BlockEqual, vhashing::std_memspace>
        // bmhi(*dev_blockmap_);
        // int count = 0;
        // for(int i = 0; i < *(bmhi.heap_counter); i ++){
        //     for(int j = 0; j < 4; j ++){
        //         int offset = i * 4 + j;
        //         vhashing::HashEntryBase<int3> &entr = bmhi.hash_table[offset];
        //         int3 ii = entr.key;
        //         if(ii.x == 999999 && ii.y == 999999 && ii.z == 999999 )
        //             continue;
        //         else{
        //             count ++;
        //         }
        //     }
        // }
        // printf("* get in camera counter %d \n", *(bmhi.heap_counter));

        // int recount = 0;
        // // stream in
        // vhashing::HashTable<int3, VoxelBlock, BlockHasher, BlockEqual, vhashing::std_memspace>
        // bmhhh(100001, 4, 400000, int3{999999, 999999, 999999});

        // // check
        // for (int i=0; i<*(bmhi.heap_counter); i++) {
        //     int3 key = bmhi.key_heap[i];
        //     // printf("key\t%d\t%d\t%d\n", key.x, key.y, key.z);
        //     if(bmhhh.find(key) == bmhhh.end()){
        //         VoxelBlock vb;
        //         bmhhh[key] = vb;
        //         recount ++;
        //     }
        // }

        // printf("count out ==== %d\n", recount);

        int *HeapCounter;
        cudaSafeCall(cudaMalloc((void**)&HeapCounter, sizeof(int)));
        
        {
            getHeapCounterKernel <<< 1, 1 >>> (HeapCounter, *dev_blockmap_);

        }

        // cudaSafeCall(cudaDeviceSynchronize()); //debug
        int dst_insert = 0;
        cudaMemcpy(&dst_insert, HeapCounter, sizeof(int), cudaMemcpyDeviceToHost);

        // unsigned int dst_u = dst_insert;
        cudaMemcpy(d_heapBlockCounter, &dst_insert, sizeof(unsigned int), cudaMemcpyHostToDevice);

        // int src = 0;
        // cudaMemcpy(HeapCounter, &src, sizeof(int), cudaMemcpyHostToDevice);
        // printf("finish HashAssignKernel | d_heapBlockCounter\t%d\n", dst_insert);

        cudaFree(HeapCounter);

        // vhashing::HashTable<int3, VoxelBlock, BlockHasher, BlockEqual, vhashing::std_memspace>
        // bmhi(*dev_blockmap_);
        // int count = 0;
        // int error = 0;
        // for(int i = 0; i < 100001; i ++){
        //     for(int j = 0; j < 4; j ++){
        //         int offset = i * 4 + j;
        //         vhashing::HashEntryBase<int3> &entr = bmhi.hash_table[offset];
        //         int3 ii = entr.key;
        //         if(ii.x == 999999 && ii.y == 999999 && ii.z == 999999 )
        //             continue;
        //         else{

        //             if(bmhi_chunk.find(entr.key) == bmhi_chunk.end())
        //                 error ++;
        //             count ++;
        //         }
        //     }
        // }

        // {
        //                             // d_inBlockPosHeap[addr].pos = idCurrentBlock;
        //                 // uint addr = atomicAdd(&d_heapBlockCounter[0], 1);
        // }
        // printf("error %d count %d in map\n",error,count);
    }
}



