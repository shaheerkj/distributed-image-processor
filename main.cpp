#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <algorithm>
#include <iomanip>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

// Image class with stb_image support
class Image {
public:
    int width, height, channels;
    std::vector<unsigned char> data;

    Image(int w = 0, int h = 0, int c = 3) 
        : width(w), height(h), channels(c), data(w * h * c) {}

    unsigned char& at(int x, int y, int c) {
        return data[(y * width + x) * channels + c];
    }

    const unsigned char& at(int x, int y, int c) const {
        return data[(y * width + x) * channels + c];
    }

    static Image load(const std::string& filename) {
        int width, height, channels;
        unsigned char* img_data = stbi_load(filename.c_str(), &width, &height, &channels, 0);
        
        if (!img_data) {
            std::cerr << "Failed to load image: " << filename << std::endl;
            std::cerr << "Error: " << stbi_failure_reason() << std::endl;
            return Image();
        }

        Image img(width, height, channels);
        img.data.assign(img_data, img_data + (width * height * channels));
        
        stbi_image_free(img_data);
        return img;
    }

    bool save(const std::string& filename) const {
        // Determine format from extension
        std::string ext = filename.substr(filename.find_last_of(".") + 1);
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

        int result = 0;
        if (ext == "png") {
            result = stbi_write_png(filename.c_str(), width, height, channels, 
                                   data.data(), width * channels);
        } else if (ext == "jpg" || ext == "jpeg") {
            result = stbi_write_jpg(filename.c_str(), width, height, channels, 
                                   data.data(), 90);
        } else if (ext == "bmp") {
            result = stbi_write_bmp(filename.c_str(), width, height, channels, 
                                   data.data());
        } else {
            std::cerr << "Unsupported output format: " << ext << std::endl;
            return false;
        }

        return result != 0;
    }
};

// ---------------- Filters -----------------

class ImageFilters {
public:

    static void grayscale(Image& img) {
        #pragma omp parallel for
        for (int i = 0; i < img.width * img.height; i++) {
            int idx = i * img.channels;
            unsigned char r = img.data[idx];
            unsigned char g = img.data[idx + 1];
            unsigned char b = img.data[idx + 2];
            unsigned char gray = (unsigned char)(0.299 * r + 0.587 * g + 0.114 * b);
            img.data[idx] = img.data[idx + 1] = img.data[idx + 2] = gray;
        }
    }

    static void blur(Image& img, int radius = 2) {
        Image temp = img;

        #pragma omp parallel for collapse(2)
        for (int y = 0; y < img.height; y++) {
            for (int x = 0; x < img.width; x++) {
                for (int c = 0; c < img.channels; c++) {
                    int sum = 0, count = 0;

                    for (int dy = -radius; dy <= radius; dy++) {
                        for (int dx = -radius; dx <= radius; dx++) {
                            int nx = x + dx;
                            int ny = y + dy;
                            if (nx >= 0 && nx < img.width && ny >= 0 && ny < img.height) {
                                sum += temp.at(nx, ny, c);
                                count++;
                            }
                        }
                    }
                    img.at(x, y, c) = (unsigned char)(sum / count);
                }
            }
        }
    }

    static void sharpen(Image& img) {
        Image temp = img;
        const int kernel[3][3] = {
            { 0, -1,  0},
            {-1,  5, -1},
            { 0, -1,  0}
        };

        #pragma omp parallel for collapse(2)
        for (int y = 1; y < img.height - 1; y++) {
            for (int x = 1; x < img.width - 1; x++) {
                for (int c = 0; c < img.channels; c++) {
                    int sum = 0;
                    for (int ky = 0; ky < 3; ky++) {
                        for (int kx = 0; kx < 3; kx++) {
                            sum += kernel[ky][kx] * temp.at(x + kx - 1, y + ky - 1, c);
                        }
                    }
                    img.at(x, y, c) = (unsigned char)std::min(255, std::max(0, sum));
                }
            }
        }
    }

    static void edgeDetect(Image& img, int sensitivity = 1) {
        Image temp = img;

        const int sobelX[3][3] = {
            {-1, 0, 1},
            {-2, 0, 2},
            {-1, 0, 1}
        };

        const int sobelY[3][3] = {
            {-1, -2, -1},
            { 0,  0,  0},
            { 1,  2,  1}
        };

        #pragma omp parallel for collapse(2)
        for (int y = 1; y < img.height - 1; y++) {
            for (int x = 1; x < img.width - 1; x++) {
                for (int c = 0; c < img.channels; c++) {

                    int gx = 0, gy = 0;

                    for (int ky = 0; ky < 3; ky++) {
                        for (int kx = 0; kx < 3; kx++) {
                            int pixel = temp.at(x + kx - 1, y + ky - 1, c);
                            gx += sobelX[ky][kx] * pixel;
                            gy += sobelY[ky][kx] * pixel;
                        }
                    }

                    int magnitude = (int)(sqrt(gx * gx + gy * gy) * sensitivity);
                    magnitude = std::min(255, magnitude);

                    img.at(x, y, c) = (unsigned char)magnitude;
                }
            }
        }
    }
};

// ----------------- Main ---------------------

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    std::string input_image_path;
    std::string output_image_path;
    int blur_radius = 2;
    int apply_sharpen = 1;
    int edge_sensitivity = 1;

    // ---------------- User Input (Root Only) -----------------

    if (world_rank == 0) {
        std::cout << "Enter image path (JPG/PNG/BMP): ";
        std::cin >> input_image_path;

        std::cout << "Enter output path (e.g., output.png): ";
        std::cin >> output_image_path;

        std::cout << "Enter blur radius (0–10): ";
        std::cin >> blur_radius;
        blur_radius = std::max(0, std::min(10, blur_radius));

        std::cout << "Apply sharpen? (1 yes, 0 no): ";
        std::cin >> apply_sharpen;

        std::cout << "Edge detection sensitivity (1–5): ";
        std::cin >> edge_sensitivity;
        edge_sensitivity = std::max(1, std::min(5, edge_sensitivity));
    }

    // -------------- Broadcast user inputs ----------------

    MPI_Bcast(&blur_radius, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&apply_sharpen, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&edge_sensitivity, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int path_len = input_image_path.size();
    MPI_Bcast(&path_len, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (world_rank != 0)
        input_image_path.resize(path_len);

    MPI_Bcast(input_image_path.data(), path_len, MPI_CHAR, 0, MPI_COMM_WORLD);

    int out_path_len = output_image_path.size();
    MPI_Bcast(&out_path_len, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (world_rank != 0)
        output_image_path.resize(out_path_len);

    MPI_Bcast(output_image_path.data(), out_path_len, MPI_CHAR, 0, MPI_COMM_WORLD);

    // ---------------- Load Image (Root Only) -----------------

    Image fullImage;
    int IMG_WIDTH = 0, IMG_HEIGHT = 0, CHANNELS = 3;

    if (world_rank == 0) {
        fullImage = Image::load(input_image_path);

        if (fullImage.width == 0) {
            std::cerr << "Could not load input image.\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        IMG_WIDTH = fullImage.width;
        IMG_HEIGHT = fullImage.height;
        CHANNELS = fullImage.channels;

        std::cout << "\n=== Image Loaded ===\n";
        std::cout << "Size: " << IMG_WIDTH << "x" << IMG_HEIGHT << "\n";
        std::cout << "Channels: " << CHANNELS << "\n";
    }

    // Broadcast image dimensions to all processes
    MPI_Bcast(&IMG_WIDTH, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&IMG_HEIGHT, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&CHANNELS, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // ---------------- Split Work -----------------

    int rows_per_process = IMG_HEIGHT / world_size;
    int extra_rows = IMG_HEIGHT % world_size;

    int my_rows = rows_per_process + (world_rank < extra_rows ? 1 : 0);
    int my_start_row = world_rank * rows_per_process + std::min(world_rank, extra_rows);

    std::vector<int> sendcounts(world_size);
    std::vector<int> displs(world_size);

    if (world_rank == 0) {
        int offset = 0;
        for (int i = 0; i < world_size; i++) {
            int rows = rows_per_process + (i < extra_rows ? 1 : 0);
            sendcounts[i] = rows * IMG_WIDTH * CHANNELS;
            displs[i] = offset;
            offset += sendcounts[i];
        }
    }

    std::vector<unsigned char> local_chunk(my_rows * IMG_WIDTH * CHANNELS);

    MPI_Scatterv(
        world_rank == 0 ? fullImage.data.data() : nullptr,
        sendcounts.data(),
        displs.data(),
        MPI_UNSIGNED_CHAR,
        local_chunk.data(),
        local_chunk.size(),
        MPI_UNSIGNED_CHAR,
        0,
        MPI_COMM_WORLD
    );

    Image localImage(IMG_WIDTH, my_rows, CHANNELS);
    localImage.data = local_chunk;

    // ---------------- Process (OpenMP) --------------------

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            // optional grayscale section
            Image temp = localImage;
            ImageFilters::grayscale(temp);
        }

        #pragma omp section
        {
            ImageFilters::blur(localImage, blur_radius);
        }
    }

    if (apply_sharpen)
        ImageFilters::sharpen(localImage);

    ImageFilters::edgeDetect(localImage, edge_sensitivity);

    // ---------------- Gather result ---------------------

    MPI_Request gather_request;

    MPI_Igatherv(
        localImage.data.data(),
        localImage.data.size(),
        MPI_UNSIGNED_CHAR,
        world_rank == 0 ? fullImage.data.data() : nullptr,
        sendcounts.data(),
        displs.data(),
        MPI_UNSIGNED_CHAR,
        0,
        MPI_COMM_WORLD,
        &gather_request
    );

    MPI_Wait(&gather_request, MPI_STATUS_IGNORE);

    if (world_rank == 0) {
        if (fullImage.save(output_image_path)) {
            std::cout << "\n=== Processing Complete ===\n";
            std::cout << "Output saved as: " << output_image_path << "\n";
        } else {
            std::cerr << "Failed to save output image.\n";
        }
    }

    MPI_Finalize();
    return 0;
} 
