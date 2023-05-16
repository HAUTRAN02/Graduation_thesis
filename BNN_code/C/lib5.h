/**
 * @file lib5.h
 * @brief Prototypes, common definitions  for Component pred(BNN_lenet5)
 *
 * This file contains the function prototypes, common definitions , constants, and data structures
 * related to the firmware/driver for pred(BNN_lenet5). It provides the interface
 * for initializing, configuring, and controlling the Component pred(BNN_lenet5). 
 *
 * @author
 * Hau Tran <real@example.com>
 *
 * @version 1.0
 *
 * @date 2023-05-16
 */
#include <stdint.h>
#include <stdlib.h>
#include <math.h>

#define LABEL_LEN 100

void pred(float *image,
                    int *w_conv1,
                    int *b_conv1,
                    int *w_conv2,
                    int *b_conv2,
                    int *w_fc1,
                    int *b_fc1,
                    int *w_fc2,
                    int *b_fc2,
                    float *w_fc3,
                    float *b_fc3,
                    int *result
                    );

