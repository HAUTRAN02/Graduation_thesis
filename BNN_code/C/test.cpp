//  Copyright (c) 2021 Intel Corporation
//  SPDX-License-Identifier: MIT

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

  int o_conv1c[6 * 24 * 24];
  int o_avgpooling1c[6 * 12 * 12];
  int o_conv2c[16 * 8 * 8];
  int o_avgpooling2c[16 * 4 * 4];
  int o_fc1c[120];
  int o_fc2c[84];
  float o_fc3c[10];

  int diff = 0;

  int o_conv1[6 * 24 * 24];
  int o_avgpooling1[6 * 12 * 12];
  int o_conv2[16 * 8 * 8];
  int o_avgpooling2[16 * 4 * 4];
  int o_fc1[120];
  int o_fc2[84];
  float o_fc3[10];

  float temp2;
  float temp;

  /** START TEST **/
  FILE *fp;
  fp = fopen("../data/weight2/out_conv1.txt", "r");
  for (int channel = 0; channel < 6; channel++)
  {
    for (int row = 0; row < 24; row++)
    {
      for (int col = 0; col < 24; col++)
      {
        fscanf(fp, "%f ", &(temp2));
        o_conv1[24 * 24 * channel + 24 * row + col] = (int)temp2;
        
      }
    }
  }

  fclose(fp);

  fp = fopen("../data/weight2/out_pool.txt", "r");
  int t1;
  for (int n_channel = 0; n_channel < 6; n_channel++)
  {
    for (i = 0; i < 24; i += 2)
    {
      for (j = 0; j < 24; j += 2)
      {
        fscanf(fp, "%f ", &(temp2));
        t1 = n_channel * 12 * 12 + 12 * (i / 2) + (j / 2);
        o_avgpooling1[t1] = (int)temp2;
      }
    }
  }
  fclose(fp);

  fp = fopen("../data/weight2/out_conv2.txt", "r");
  for (int channel = 0; channel < 16; channel++)
  {
    for (int row = 0; row < 8; row++)
    {
      for (int col = 0; col < 8; col++)
      {
        fscanf(fp, "%f ", &(temp2));
        o_conv2[channel * 8 * 8 + 8 * row + col] = (int)temp2;
      }
    }
  }
  fclose(fp);

  fp = fopen("../data/weight2/out_pool2.txt", "r");
  int t2 = 0;
  for (int n_channel = 0; n_channel < 16; n_channel++)
  {
    for (int i = 0; i < 8; i += 2)
    {
      for (int j = 0; j < 8; j += 2)
      {
        t2 = n_channel * 4 * 4 + 4 * (i / 2) + (j / 2);
        fscanf(fp, "%f ", &(temp2));
        o_avgpooling2[t2] = (int)temp2;
      }
    }
  }
  fclose(fp);

  fp = fopen("../data/weight2/out_fc1.txt", "r");
  for (int i = 0; i < 120; i++)
  {
    for (int j = 0; j < 256; j++)
    {
      fscanf(fp, "%f ", &(temp2));
      o_fc1[i] = (int)temp2;
    }
  }

  fp = fopen("../data/weight2/out_fc2.txt", "r");
  for (int i = 0; i < 84; i++)
  {
    for (int j = 0; j < 120; j++)
    {
      fscanf(fp, "%f ", &(temp2));
      o_fc2[i] = (int)temp2;
    }
  }
  fclose(fp);

  fp = fopen("../data/weight2/out_fc3.txt", "r");
  for (int i = 0; i < 10; i++)
  {
    for (int j = 0; j < 84; j++)
    {
      fscanf(fp, "%f ", &(temp2));
      o_fc3[i] = temp2;
    }
  }
  fclose(fp);

  /** END Test **/

  fp = fopen("../data/weights/conv1.weight.txt", "r");
  for (int j = 0; j < 6; j++)
  {
    for (int m = 0; m < 5; m++)
    {
      for (int n = 0; n < 5; n++)
      {
        fscanf(fp, "%f ", &(temp));
        // printf("%d \t",temp);
        // w_conv1[5 * 5 * j + 5 * m + n] = temp;
        w_conv1[5 * 5 * j + 5 * m + n] = (int)temp;
        // printf("%d \t",(int)w_conv1[5 * 5 * j + 5 * m + n]);
      }
    }
  }

  fclose(fp);

  fp = fopen("../data/weights/conv2.weight.txt", "r");
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

  fp = fopen("../data/weights/fc1.weight.txt", "r");
  for (i = 0; i < 120; i++)
  {
    for (j = 0; j < 256; j++)
    {
      fscanf(fp, "%f", &(temp));
      w_fc1[256 * i + j] = (int)temp;
    }
  }
  fclose(fp);

  fp = fopen("../data/weights/fc2.weight.txt", "r");
  for (i = 0; i < 84; i++)
  {
    for (j = 0; j < 120; j++)
    {
      fscanf(fp, "%f ", &(temp));
      w_fc2[120 * i + j] = (int)temp;
    }
  }
  fclose(fp);

  fp = fopen("../data/weights/fc3.weight.txt", "r");
  for (i = 0; i < 10; i++)
  {
    for (j = 0; j < 84; j++)
    {
      fscanf(fp, "%f ", &(temp));
      w_fc3[84 * i + j] = temp;
    }
  }
  fclose(fp);

  fp = fopen("../data/weights/conv1.bias.txt", "r");
  for (int i = 0; i < 6; i++)
  {
    fscanf(fp, "%f ", &(temp));
    b_conv1[i] = (int)temp;
  }

  fclose(fp);

  fp = fopen("../data/weights/conv2.bias.txt", "r");
  for (i = 0; i < 16; i++)
  {
    fscanf(fp, "%f ", &(temp));
    b_conv2[i] = (int)temp;
  }

  fclose(fp);

  fp = fopen("../data/weights/fc1.bias.txt", "r");
  for (i = 0; i < 120; i++)
  {
    fscanf(fp, "%f", &(temp));
    b_fc1[i] = (int)temp;
  }
  fclose(fp);

  fp = fopen("../data/weights/fc2.bias.txt", "r");
  for (i = 0; i < 84; i++)
  {
    fscanf(fp, "%f ", &(temp));
    b_fc2[i] = (int)temp;
  }
  fclose(fp);

  fp = fopen("../data/weights/fc3.bias.txt", "r");
  for (i = 0; i < 10; i++)
  {
    fscanf(fp, "%f ", &(temp));
    b_fc3[i] = temp;
  }

  fclose(fp);

  int target[LABEL_LEN];

  fp = fopen("../data/MNIST/mnist-test-target.txt", "r");
  for (i = 0; i < LABEL_LEN; i++)
    fscanf(fp, "%d ", &(target[i]));
  fclose(fp);

  ///////

  fp = fopen("../data/MNIST/image.txt", "r");
  for (int i = 0; i < LABEL_LEN * 28 * 28; i++)
  {

    fscanf(fp, "%f ", &(temp));
    dataset[i] = temp;
    
  }
  fclose(fp);

  pred(dataset, w_conv1, b_conv1, w_conv2, b_conv2, w_fc1, b_fc1, w_fc2, b_fc2, w_fc3, b_fc3, test_value, o_conv1c, o_avgpooling1c, o_conv2c, o_avgpooling2c, o_fc1c, o_fc2c, o_fc3c);

  test_array_int(o_conv1, o_conv1c, 24 * 24 * 6, &diff);
  //  test_array_int(o_avgpooling1, o_avgpooling1c, 12*12*6, &diff);
  // for(int dm = 0;dm<100;dm++)
  // printf("py %d vs  c %d\n",o_conv1[0],o_conv1c[0]);
  printf("\n diff %d \n", diff);

  for (i = 0; i < LABEL_LEN; i++)
  {
    if (test_value[i] == target[i])
      acc++;
    printf("Predicted label: %d\n", test_value[i]);
    printf(" label: %d\n", target[i]);
    printf("Prediction: %d/%d\n", acc, i + 1);
  }
}
