#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>
#include <stdbool.h>

#include "test_images.h"

typedef uint8_t u8;

// possible modes: 88, 17
#define mode 17

typedef short int DATA;

#define FIXED2FLOAT(a, qf) (((float)(a)) / (1 << qf))
#define FLOAT2FIXED(a, qf) ((short int)round((a) * (1 << qf)))
#define _MAX_ (1 << (sizeof(DATA) * 8 - 1)) - 1
#define _MIN_ -(_MAX_ + 1)

#define DEBUG_VERBOSE 0

#define path "aes_mnist_assigment_groups/group_4/weights"

#define img_width 28
#define img_heigh 28
#define img_size img_heigh *img_width
#define n_bias0 64
#define n_weights0 50176
#define n_bias1 32
#define n_weights1 2048
#define n_bias2 16
#define n_weights2 512
#define n_bias3 10
#define n_weights3 160

// WEIGHTS AND BIASES

#if mode == 88
DATA gemm0_bias[n_bias0];
DATA gemm0_weights[n_weights0];
DATA gemm1_bias[n_bias1];
DATA gemm1_weights[n_weights1];
DATA gemm2_bias[n_bias2];
DATA gemm2_weights[n_weights2];
DATA gemm3_bias[n_bias3];
DATA gemm3_weights[n_weights3];
#elif mode == 17
int8_t gemm0_bias[n_bias0];
int8_t gemm0_weights[n_weights0];
int8_t gemm1_bias[n_bias1];
int8_t gemm1_weights[n_weights1];
int8_t gemm2_bias[n_bias2];
int8_t gemm2_weights[n_weights2];
int8_t gemm3_bias[n_bias3];
int8_t gemm3_weights[n_weights3];
#endif

// TEST IMAGES

DATA images[10][28 * 28] = {{imm_test_0}, {imm_test_1}, {imm_test_2}, {imm_test_3}, {imm_test_4}, {imm_test_5}, {imm_test_6}, {imm_test_7}, {imm_test_8}, {imm_test_9}};

// FUNCTIONS

static inline void relu_forward(DATA *input, DATA *output, int size);
#if mode == 88
void FC_forward(DATA *input, DATA *output, int in_s, int out_s, DATA *weights, DATA *bias, int qf);
#elif mode == 17
void FC_forward(DATA *input, DATA *output, int in_s, int out_s, int8_t *weights, int8_t *bias, int qf);
#endif
int resultsProcessing(DATA *results, int size);
static inline long long int saturate(long long int mac);

uint64_t extract_bits(uint64_t source, int left_offset, int n_bits, bool shift_to_right)
{
    unsigned mask = ((1 << n_bits) - 1) << left_offset;
    uint64_t isolatedXbits = source & mask;
    if (shift_to_right)
    {
        isolatedXbits = isolatedXbits >> left_offset;
    }
    return isolatedXbits;
}

uint64_t set_bit(uint64_t source, int position, bool value)
{
    unsigned mask = 1 << position;
    if (value) return source | mask;
    return source & (~mask);
}

bool get_bit(uint64_t source, int position)
{
    unsigned mask = 1 << position;
    uint64_t isolatedbit = source & mask;
    return isolatedbit != 0;
}

#if mode == 88
int read_DATA(DATA *value, int fd)
{ // read 2 bytes from the file descriptor, return -1 if reach end of file, set value if success
    u8 data[2];
    for (int i = 0; i < 2; i++)
    {
        int result = read(fd, data + i, 1);
        if (result < 0)
        {
            return result;
        }
        else
        {
            if (result == 1)
            {
                if (DEBUG_VERBOSE > 10)
                    printf("\nRead value: %d, returned value %d", data[i], result);
            }
            else
                return -1; // end of file reached
        }
    }
    *value = (DATA)(data[1] << 8) + data[0];
    return 0;
}
#endif

#if mode == 17
int read_int16_to_int8(int8_t *value, int fd)
{
    u8 data[2];
    for (int i = 0; i < 2; i++)
    {
        int result = read(fd, data + i, 1);
        if (result < 0)
        {
            return result;
        }
        else
        {
            if (result == 1)
            {
                if (DEBUG_VERBOSE > 10)
                    printf("\nRead value: %d, returned value %d", data[i], result);
            }
            else
                return -1; // end of file reached
        }
    }
    *value = data[0];
    return 0;
}
#endif

#if mode == 88
int read_bytes_from_path(DATA *buffer, char *path_val, int to_read)
{
    int fd = open(path_val, O_RDONLY);
    if (fd < 0)
        return fd;
    int result = 0;
    for (int i = 0; i < (to_read); i++)
    {

        int c_result = read_DATA(buffer + i, fd);
        if (c_result < 0)
        {
            result = c_result;
            break;
        }
    }
    close(fd);
    return result;
}
#elif mode == 17
int read_bytes_from_path(int8_t *buffer, char *path_val, int to_read)
{
    int fd = open(path_val, O_RDONLY);
    if (fd < 0)
        return fd;
    int result = 0;
    for (int i = 0; i < (to_read); i++)
    {
        int c_result = read_int16_to_int8(buffer + i, fd);
        if (c_result < 0)
        {
            result = c_result;
            break;
        }
    }
    close(fd);
    return result;
}
#endif

void print_data_toscrr(DATA *buffer, int size)
{
    for (int i = 0; i < size; i++)
    {
        printf("%d ", buffer[i]);
    }
}

void print_data(char *title, DATA *buffer, int size)
{
    printf("\n--- %s ---\n", title);
    print_data_toscrr(buffer, size);
    printf("\n--- \n");
}

#if mode == 17
void print_data_toscrr_int8(int8_t *buffer, int size)
{
    for (int i = 0; i < size; i++)
    {
        printf("%d ", buffer[i]);
    }
}

void print_data_int8(char *title, int8_t *buffer, int size)
{
    printf("\n--- %s ---\n", title);
    print_data_toscrr_int8(buffer, size);
    printf("\n--- \n");
}
#endif

int store_image_to_file(DATA *image, char *path_val)
{
    int fd = open(path_val, O_WRONLY | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR);
    if (fd < 0)
        return fd;
    int result = 0;
    for (int i = 0; i < img_size; i++)
    {
        if (write(fd, image + i, sizeof(DATA)) < sizeof(DATA))
        {
            result = -1;
            break;
        }
    }
    close(fd);
    return result;
}

int main()
{
    // DATA image[img_size];
    DATA output_gemm0[n_bias0];
    DATA input_gemm1[n_bias0];
    DATA output_gemm1[n_bias1];
    DATA input_gemm2[n_bias1];
    DATA output_gemm2[n_bias2];
    DATA input_gemm3[n_bias2];
    DATA output_gemm3[n_bias3];

    if (read_bytes_from_path(gemm0_bias, "aes_mnist_assigment_groups/group_4/weights/Gemm0_biases.bin", n_bias0) < 0)
        return -1;
    if (read_bytes_from_path(gemm1_bias, "aes_mnist_assigment_groups/group_4/weights/Gemm1_biases.bin", n_bias1) < 0)
        return -1;
    if (read_bytes_from_path(gemm2_bias, "aes_mnist_assigment_groups/group_4/weights/Gemm2_biases.bin", n_bias2) < 0)
        return -1;
    if (read_bytes_from_path(gemm3_bias, "aes_mnist_assigment_groups/group_4/weights/Gemm3_biases.bin", n_bias3) < 0)
        return -1;

    if (read_bytes_from_path(gemm0_weights, "aes_mnist_assigment_groups/group_4/weights/Gemm0_weights.bin", n_weights0) < 0)
        return -1;
    if (read_bytes_from_path(gemm1_weights, "aes_mnist_assigment_groups/group_4/weights/Gemm1_weights.bin", n_weights1) < 0)
        return -1;
    if (read_bytes_from_path(gemm2_weights, "aes_mnist_assigment_groups/group_4/weights/Gemm2_weights.bin", n_weights2) < 0)
        return -1;
    if (read_bytes_from_path(gemm3_weights, "aes_mnist_assigment_groups/group_4/weights/Gemm3_weights.bin", n_weights3) < 0)
        return -1;

    

#if mode == 88
    int qf_v = 8;
    for (int i = 0; i < 10; i++)
    {
        FC_forward(images[i], output_gemm0, img_size, n_bias0, gemm0_weights, gemm0_bias, qf_v);
        relu_forward(output_gemm0, input_gemm1, n_bias0);
        FC_forward(input_gemm1, output_gemm1, n_bias0, n_bias1, gemm1_weights, gemm1_bias, qf_v);
        relu_forward(output_gemm1, input_gemm2, n_bias1);
        FC_forward(input_gemm2, output_gemm2, n_bias1, n_bias2, gemm2_weights, gemm2_bias, qf_v);
        relu_forward(output_gemm2, input_gemm3, n_bias2);
        FC_forward(input_gemm3, output_gemm3, n_bias2, n_bias3, gemm3_weights, gemm3_bias, qf_v);

        resultsProcessing(output_gemm3, 10);
    }
#elif mode == 17
    int qf_v = 7;
    for (int i = 0; i < 10; i++)
    {
        FC_forward(images[i], output_gemm0, img_size, n_bias0, gemm0_weights, gemm0_bias, qf_v);
        relu_forward(output_gemm0, input_gemm1, n_bias0);
        FC_forward(input_gemm1, output_gemm1, n_bias0, n_bias1, gemm1_weights, gemm1_bias, qf_v);
        relu_forward(output_gemm1, input_gemm2, n_bias1);
        FC_forward(input_gemm2, output_gemm2, n_bias1, n_bias2, gemm2_weights, gemm2_bias, qf_v);
        relu_forward(output_gemm2, input_gemm3, n_bias2);
        FC_forward(input_gemm3, output_gemm3, n_bias2, n_bias3, gemm3_weights, gemm3_bias, qf_v);

        resultsProcessing(output_gemm3, 10);
    }
#endif

    return 0;
}

#define SIZEWA 10

#if mode == 88
void FC_forward(DATA *input, DATA *output, int in_s, int out_s, DATA *weights, DATA *bias, int qf)
{ // NOTE return W * x
    int hkern = 0;
    int wkern = 0;
    long long int mac = 0; // 64 bits
    DATA current = 0;
    /* foreach row in kernel */
    //	#pragma omp parallel for private (hkern, wkern, mac, current)
    for (hkern = 0; hkern < out_s; hkern++)
    {
        mac = ((long long int)bias[hkern]) << qf;
        for (wkern = 0; wkern < in_s; wkern++)
        {
            current = input[wkern];
            mac += current * weights[hkern * in_s + wkern]; // matrix, element in position hkern, wkern
        }
        output[hkern] = (DATA)saturate(mac >> qf);
    }
}
#elif mode == 17
void FC_forward(DATA *input, DATA *output, int in_s, int out_s, int8_t *weights, int8_t *bias, int qf)
{ // NOTE return W * x
    int hkern = 0;
    int wkern = 0;
    long long int mac = 0; // 64 bits
    int8_t current = 0;
    /* foreach row in kernel */
    //	#pragma omp parallel for private (hkern, wkern, mac, current)
    for (hkern = 0; hkern < out_s; hkern++)
    {
        mac = ((long long int)bias[hkern]) << qf;
        for (wkern = 0; wkern < in_s; wkern++)
        {
            current = input[wkern];
            mac += current * weights[hkern * in_s + wkern]; // matrix, element in position hkern, wkern
        }
        output[hkern] = (DATA)saturate(mac >> qf);
    }
}
#endif

static inline void relu_forward(DATA *input, DATA *output, int size)
{
    int i = 0;
    for (i = 0; i < size; i++)
    {
        DATA v = input[i];
        v = v > 0 ? v : 0;
        output[i] = v;
    }
}

int resultsProcessing(DATA *results, int size)
{ // What do you want to do with the results of the CNN? Here is the place where you should put the classifier or the detection (see YOLO detection for example)
    // The simplest classifier is a maximum search for the results which returns the index value of the maximum
    // char *labels[10] = {"digit 0", "digit 1", "digit 2", "digit 3", "digit 4", "digit 5", "digit 6", "digit 7", "digit 8", "digit 9"};
    // TODO: check the size parameter
    int size_wa = SIZEWA;
    float r[SIZEWA];
    int c[SIZEWA];
    float results_float[SIZEWA];
    float sum = 0.0;
    DATA max = 0;
    int max_i;
    for (int i = 0; i < size_wa; i++)
    {
        results_float[i] = FIXED2FLOAT(results[i], 8);
        int n;
        if (results[i] > 0)
            n = results[i];
        else
            n = -results[i];
        if (n > max)
        {
            max = n;
            max_i = i;
        }
    }
    for (int i = 0; i < size_wa; i++)
        sum += exp(results_float[i]);
    for (int i = 0; i < size_wa; i++)
    {
        r[i] = exp(results_float[i]) / sum;
        c[i] = i;
    }
    for (int i = 0; i < size_wa; i++)
    {
        for (int j = i; j < size_wa; j++)
        {
            if (r[j] > r[i])
            {
                float t = r[j];
                r[j] = r[i];
                r[i] = t;
                int tc = c[j];
                c[j] = c[i];
                c[i] = tc;
            }
        }
    }
    int top0 = 0;
    float topval = results_float[0];
    for (int i = 1; i < size_wa; i++)
    {
        if (results_float[i] > topval)
        {
            top0 = i;
            topval = results_float[i];
        }
    }
    printf("\n\n");
    for (int i = 0; i < 5; i++)
    {
        // xil_printf("            TOP %d: [%d] %s   \n", i, c[i], labels[c[i]]);
    }
    printf("max= %x \n", top0);
    return top0;
}

static inline long long int saturate(long long int mac)
{
    if (mac > _MAX_)
    {
        if (DEBUG_VERBOSE > 1)
            printf("[WARNING] Saturation.mac: %lld -> %llx _MAX_: %d  _MIN_: %d  res: %d\n", mac, mac, _MAX_, _MIN_, _MAX_);
        return _MAX_;
    }
    if (mac < _MIN_)
    {
        if (DEBUG_VERBOSE > 1)
            printf("[WARNING] Saturation. mac: %lld -> %llx _MAX_: %d  _MIN_: %d  res: %d\n", mac, mac, _MAX_, _MIN_, _MIN_);
        return _MIN_;
    } // printf("mac: %lld -> %llx _MAX_: %lld  _MIN_: %lld  res: %lld\n", mac, mac, _MAX_, _MIN_, mac);
    return mac;
}