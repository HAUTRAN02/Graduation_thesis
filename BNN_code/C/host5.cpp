/**
 * @file host5.cpp
 * @brief Demo Program for BNN_lenet5
 *
 * This program demonstrates the usage of the firmware/driver for Component pred(BNN_lenet5).
 * It includes function calls to initialize, configure, and control the Component
 * pred(BNN_lenet5). The program also showcases the usage of other utility functions related
 * to the Component pred(BNN_lenet5). It provides a basic example of how to interact with the
 * Component pred(BNN_lenet5) using the provided firmware/driver.
 *
 * @author
 * Hau Tran <real@example.com>
 *
 * @version 1.0
 *
 * @date 2023-05-16
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "lib5.cpp"

int main()
{
  // float *dataset = (float *)malloc(LABEL_LEN * 28 * 28 * sizeof(float));
  float dataset[LABEL_LEN * 28 * 28];
  int i, j, k, m, n, index;
  int acc = 0;
  int mm, nn;
  float *datain;
  int image[28 * 28];
  int w_conv1[6 * 5 * 5];
  int w_conv2[16 * 6 * 5 * 5];
  int w_fc1[120 * 256];
  int w_fc2[84 * 120];
  float w_fc3[10 * 84];
  int b_conv1[6];
  int b_conv2[16];
  int b_fc1[120];
  int b_fc2[84];
  float b_fc3[10];
  int probs[10];
  int test_value[LABEL_LEN];
  float temp;

  /** Read the information of weights, bias, and data of the image into an array **/
  FILE *fp;

  fp = fopen("C:/Users/Admins/Downloads/Seminar/BNN_code/data/weights/conv1.weight.txt", "r");
  for (int j = 0; j < 6; j++)
  {
    for (int m = 0; m < 5; m++)
    {
      for (int n = 0; n < 5; n++)
      {
        fscanf(fp, "%f ", &(temp));

        w_conv1[5 * 5 * j + 5 * m + n] = (int)temp;
      }
    }
  }
  fclose(fp);

  fp = fopen("C:/Users/Admins/Downloads/Seminar/BNN_code/data/weights/conv2.weight.txt", "r");
  for (i = 0; i < 16; i++)
  {
    for (j = 0; j < 6; j++)
    {
      for (m = 0; m < 5; m++)
      {
        for (n = 0; n < 5; n++)
        {
          index = 16 * i + 6 * j + 5 * m + 5 * n;
          fscanf(fp, "%f ", &(temp));
          w_conv2[6 * 5 * 5 * i + 5 * 5 * j + 5 * m + n] = (int)temp;
        }
      }
    }
  }
  fclose(fp);

  fp = fopen("C:/Users/Admins/Downloads/Seminar/BNN_code/data/weights/fc1.weight.txt", "r");
  for (i = 0; i < 120; i++)
  {
    for (j = 0; j < 256; j++)
    {
      fscanf(fp, "%f", &(temp));
      w_fc1[256 * i + j] = (int)temp;
    }
  }
  fclose(fp);

  fp = fopen("C:/Users/Admins/Downloads/Seminar/BNN_code/data/weights/fc2.weight.txt", "r");
  for (i = 0; i < 84; i++)
  {
    for (j = 0; j < 120; j++)
    {
      fscanf(fp, "%f ", &(temp));
      w_fc2[120 * i + j] = (int)temp;
    }
  }
  fclose(fp);

  fp = fopen("C:/Users/Admins/Downloads/Seminar/BNN_code/data/weights/fc3.weight.txt", "r");
  for (i = 0; i < 10; i++)
  {
    for (j = 0; j < 84; j++)
    {
      fscanf(fp, "%f ", &(temp));
      w_fc3[84 * i + j] = temp;
    }
  }
  fclose(fp);

  fp = fopen("C:/Users/Admins/Downloads/Seminar/BNN_code/data/weights/conv1.bias.txt", "r");
  for (int i = 0; i < 6; i++)
  {
    fscanf(fp, "%f ", &(temp));
    b_conv1[i] = (int)temp;
  }

  fclose(fp);

  fp = fopen("C:/Users/Admins/Downloads/Seminar/BNN_code/data/weights/conv2.bias.txt", "r");
  for (i = 0; i < 16; i++)
  {
    fscanf(fp, "%f ", &(temp));
    b_conv2[i] = (int)temp;
  }

  fclose(fp);

  fp = fopen("C:/Users/Admins/Downloads/Seminar/BNN_code/data/weights/fc1.bias.txt", "r");
  for (i = 0; i < 120; i++)
  {
    fscanf(fp, "%f", &(temp));
    b_fc1[i] = (int)temp;
  }
  fclose(fp);

  fp = fopen("C:/Users/Admins/Downloads/Seminar/BNN_code/data/weights/fc2.bias.txt", "r");
  for (i = 0; i < 84; i++)
  {
    fscanf(fp, "%f ", &(temp));
    b_fc2[i] = (int)temp;
  }
  fclose(fp);

  fp = fopen("C:/Users/Admins/Downloads/Seminar/BNN_code/data/weights/fc3.bias.txt", "r");
  for (i = 0; i < 10; i++)
  {
    fscanf(fp, "%f ", &(temp));
    b_fc3[i] = temp;
  }

  fclose(fp);

  int target[LABEL_LEN];

  fp = fopen("C:/Users/Admins/Downloads/Seminar/BNN_code/data/MNIST/mnist-test-target.txt", "r");
  for (i = 0; i < LABEL_LEN; i++)
    fscanf(fp, "%d ", &(target[i]));
  fclose(fp);

  fp = fopen("C:/Users/Admins/Downloads/Seminar/BNN_code/data/MNIST/mnist-test-image.txt", "r");
  for (int i = 0; i < LABEL_LEN * 28 * 28; i++)
  {

    fscanf(fp, "%f ", &(temp));
    dataset[i] = temp;
  }
  fclose(fp);

  /** Make inferences about the value of the image **/
  pred(dataset, w_conv1, b_conv1, w_conv2, b_conv2, w_fc1, b_fc1, w_fc2, b_fc2, w_fc3, b_fc3, test_value);
  for (i = 0; i < LABEL_LEN; i++)
  {
    if (test_value[i] == target[i])
      acc++;
    printf("Predicted label: %d\n", test_value[i]);
    printf(" label: %d\n", target[i]);
    printf("Prediction: %d/%d\n", acc, i + 1);
  }
}
