#include <iostream>
#include <fstream>
#include <vector>
#include <list>
#include <cmath>
#include <cstdlib>
#include <unordered_map>
#include <cstring>
#include <GL/glew.h>
#include <GL/glut.h>

#include "tsdf.cuh"

// CUDA kernel function to integrate a TSDF voxel volume given depth images
namespace ark {
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
        }
    }

    __host__
    GpuTsdfGenerator::GpuTsdfGenerator(int width, int height, float fx, float fy, float cx, float cy, float max_depth,
                                       float origin_x = -1.5f, float origin_y = -1.5f, float origin_z = 0.5f,
                                       float vox_size = 0.006f, float trunc_m = 0.03f, int vox_dim_x = 500,
                                       int vox_dim_y = 500, int vox_dim_z = 500) {
        im_width_ = width;
        im_height_ = height;

        memset(K_, 0.0f, sizeof(float) * 3 * 3);
        K_[0] = fx;
        K_[2] = cx;
        K_[4] = fy;
        K_[5] = cy;
        K_[8] = 1.0f;

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

        param_->vox_size = vox_size;

        param_->trunc_margin = trunc_m;

        param_->total_vox = param_->vox_dim.x * param_->vox_dim.y * param_->vox_dim.z;

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
    void GpuTsdfGenerator::processFrame(float *depth, unsigned char *rgb, float *c2w) {
        cudaMemcpy(dev_c2w_, c2w, 4 * 4 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_depth_, depth, im_height_ * im_width_ * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_rgb_, rgb, 3 * im_height_ * im_width_ * sizeof(unsigned char), cudaMemcpyHostToDevice);
        checkCUDA(__LINE__, cudaGetLastError());

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

        std::unordered_map<std::string, int> verticesIdx;
        std::vector<std::list<std::pair<Vertex, int>>> hash_table(param_->total_vox,
                                                                  std::list<std::pair<Vertex, int>>());

        int vertexCount = 0;

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
            if (param_->edgeTable[cubeIndex] == 0)
                continue;

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
}



