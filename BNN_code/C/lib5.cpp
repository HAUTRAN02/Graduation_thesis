/**
 * @file lib5.cpp
 * @brief Driver for Component BNN_lenet5
 *
 * This file contains the firmware/driver implementation for pred(BNN_lenet5).
 * It provides functions for initializing, configuring, and controlling the
 * Component pred(BNN_lenet5)
 * 
 * @note The following code performs a binary conversion of the following data: 
 * w_conv1, b_conv1, w_conv2, b_conv2, w_fc1, b_fc1, w_fc2, b_fc2, conv1_output, 
 * pool1_output, conv2_output, pool2_output, fc1_output
 *
 * @author
 * Hau Tran <trunghautran0301@gmail.com>
 *
 * @version 1.0
 *
 * @date 2023-05-16
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include "lib5.h"

void conv1(
    int *input0,
    int *w_conv1,
    int *b_conv1,
    int *o_conv1)
{
  int out_Conv1 = 0;
  int channel, row, col, i, j;
  for (channel = 0; channel < 6; channel++)
  {
    for (row = 0; row < 24; row++)
    {
      for (col = 0; col < 24; col++)
      {
        out_Conv1 = 0;
        for (i = 0; i < 5; i++)
        {
          for (j = 0; j < 5; j++)
          {

            if (i == 0 && j == 0)
              out_Conv1 = input0[(row + i) * 28 + (col + j)] * w_conv1[5 * 5 * channel + 5 * i + j];
            else
              out_Conv1 += input0[(row + i) * 28 + (col + j)] * w_conv1[5 * 5 * channel + 5 * i + j];
          }
        }
        if (out_Conv1 > 0)
        {
          o_conv1[24 * 24 * channel + 24 * row + col] = 1;
        }
        else if (out_Conv1 < 0)
        {
          o_conv1[24 * 24 * channel + 24 * row + col] = -1;
        }

        // o_conv1[24 * 24 * channel + 24 * row + col] = ((out_Conv1) > 0) ? 1 :( ((out_Conv1) < 0) ? -1 : 0);
      }
    }
  }
}

void avgpooling1(int *input1,
                 int *output1)
{
  int t1;
  float out2 = 0;
  int n_channel, i, j;
  for (n_channel = 0; n_channel < 6; n_channel++)
  {
    for (i = 0; i < 24; i += 2)
    {
      for (j = 0; j < 24; j += 2)
      {
        t1 = n_channel * 12 * 12 + 12 * (i / 2) + (j / 2);
        out2 = ((int)input1[n_channel * 24 * 24 + 24 * i + j] + (int)input1[n_channel * 24 * 24 + (i + 1) * 24 + j] + (int)input1[n_channel * 24 * 24 + i * 24 + (j + 1)] + (int)input1[n_channel * 24 * 24 + (i + 1) * 24 + (j + 1)]);
        output1[t1] = ((out2 / 4.0f) > 0) ? 1 : (((out2 / 4.0f) < 0) ? -1 : 0);
      }
    }
  }
}

void conv2(int *input2,
           int *kernel,
           int *bias,
           int *output2)
{
  int channel, row, col;
  int i, j, k;
  int out_conv2;
  for (channel = 0; channel < 16; channel++)
  {
    for (row = 0; row < 8; row++)
    {
      for (col = 0; col < 8; col++)
      {
        out_conv2 = 0;
        for (k = 0; k < 6; k++)
        {
          for (i = 0; i < 5; i++)
          {
            for (j = 0; j < 5; j++)
            {
              if (k == 0 && i == 0 && j == 0)
                out_conv2 = input2[k * 12 * 12 + 12 * (row + i) + col + j] * kernel[channel * 6 * 5 * 5 + k * 5 * 5 + 5 * i + j] + bias[channel];
              else
                out_conv2 += input2[k * 12 * 12 + 12 * (row + i) + col + j] * kernel[channel * 6 * 5 * 5 + k * 5 * 5 + 5 * i + j];
            }
          }
        }
        output2[channel * 8 * 8 + 8 * row + col] = ((out_conv2) > 0) ? 1 : (((out_conv2) < 0) ? -1 : 0);
      }
    }
  }
}

void avgpooling2(int *input4,
                 int *output4)
{
  int n_channel, i, j;
  int t1;
  float out3;
  int temp;
  for (n_channel = 0; n_channel < 16; n_channel++)
  {
    for (i = 0; i < 8; i += 2)
    {
      for (j = 0; j < 8; j += 2)
      {
        t1 = n_channel * 4 * 4 + 4 * (i / 2) + (j / 2);
        out3 = ((int)input4[n_channel * 8 * 8 + 8 * i + j] + (int)input4[n_channel * 8 * 8 + 8 * (i + 1) + j] + (int)input4[n_channel * 8 * 8 + i * 8 + (j + 1)] + (int)input4[n_channel * 8 * 8 + 8 * (i + 1) + j + 1]);
        output4[t1] = ((out3 / 4.0f) > 0) ? 1 : (((out3 / 4.0f) < 0) ? -1 : 0);
      }
    }
  }
}

void fc1(int *input6,
         int *weights,
         int *bias,
         int *output6)
{
  int i, j;
  int out4;
  for (i = 0; i < 120; i++)
  {
    out4 = 0;
    for (j = 0; j < 256; j++)
    {
      if (j == 0)
        out4 = (weights[i * 256 + j] * input6[j]) + bias[i];
      else
        out4 += weights[i * 256 + j] * input6[j];
    }
    output6[i] = (out4 > 0) ? 1 : ((out4 < 0) ? -1 : 0);
  }
}

void fc2(int *input8,
         int *weights,
         int *bias,
         int *output8)
{
  int i, j;
  for (i = 0; i < 84; i++)
  {
    output8[i] = 0;
    for (j = 0; j < 120; j++)
    {
      if (j == 0)
        output8[i] = (weights[i * 120 + j] * input8[j]) + bias[i];
      else
        output8[i] += weights[i * 120 + j] * input8[j];
    }
  }
}

void fc3(int *input10,
         float *weights,
         float *bias,
         float *output10)
{
  int i, j;
  for (i = 0; i < 10; i++)
  {
    output10[i] = 0;
    for (j = 0; j < 84; j++)
    {
      if (j == 0)
        output10[i] = (weights[i * 84 + j] * input10[j]) + bias[i];
      else
        output10[i] += (weights[i * 84 + j] * input10[j]);
    }
  }
}

void softmax(float *input11,
             float *output11)
{
  int i;
  float temp;
  float sum = 0;
  for (i = 0; i < 10; i++)
  {
    sum += exp(input11[i]);
  }

  for (i = 0; i < 10; i++)
  {
    temp = exp(input11[i]);
    output11[i] = fabs(temp / (sum * 1.0f));
  }
}

void getmax(float *input12, int *result, int position)
{
  result[position] = 0;
  float max = input12[0];
  for (int j = 1; j < 10; j++)
  {
    if (input12[j] > max)
    {
      result[position] = j;
      max = input12[j];
    }
  }
}

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
          int *result)
{
  int i, j, k, m, n, index;
  int mm, nn;

  int o_conv1[6 * 24 * 24];
  int o_avgpooling1[6 * 12 * 12];
  int o_conv2[16 * 8 * 8];
  int o_avgpooling2[16 * 4 * 4];
  int o_fc1[120];
  int o_fc2[84];
  float o_fc3[10];

  int image1[28 * 28];
  int w_conv11[6 * 5 * 5];
  int w_conv22[16 * 6 * 5 * 5];
  int w_fc11[120 * 256];
  int w_fc22[84 * 120];
  float w_fc33[10 * 84];
  int b_conv11[6];
  int b_conv22[16];
  int b_fc11[120];
  int b_fc22[84];
  float b_fc33[10];
  float probs[10];

  for (j = 0; j < 6; j++)
  {
    for (m = 0; m < 5; m++)
    {
      for (n = 0; n < 5; n++)
      {
        w_conv11[5 * 5 * j + 5 * m + n] = w_conv1[5 * 5 * j + 5 * m + n];
      }
    }
  }
  for (int i = 0; i < 6; i++)
  {
    b_conv11[i] = b_conv1[i];
  }
  for (i = 0; i < 16; i++)
  {
    for (j = 0; j < 6; j++)
    {
      for (m = 0; m < 5; m++)
      {
        for (n = 0; n < 5; n++)
        {

          w_conv22[6 * 5 * 5 * i + 5 * 5 * j + 5 * m + n] = w_conv2[6 * 5 * 5 * i + 5 * 5 * j + 5 * m + n];
        }
      }
    }
  }
  for (i = 0; i < 120; i++)
  {
    for (j = 0; j < 256; j++)
    {

      w_fc11[256 * i + j] = w_fc1[256 * i + j];
    }
  }
  for (i = 0; i < 84; i++)
  {
    for (j = 0; j < 120; j++)
    {

      w_fc22[120 * i + j] = w_fc2[120 * i + j];
    }
  }
  for (i = 0; i < 10; i++)
  {
    for (j = 0; j < 84; j++)
    {

      w_fc33[84 * i + j] = w_fc3[84 * i + j];
    }
  }
  for (i = 0; i < 16; i++)
  {

    b_conv22[i] = b_conv2[i];
  }
  for (i = 0; i < 120; i++)
  {

    b_fc11[i] = b_fc1[i];
  }
  for (i = 0; i < 84; i++)
  {

    b_fc22[i] = b_fc2[i];
  }
  for (i = 0; i < 10; i++)
  {
    b_fc33[i] = b_fc3[i];
  }

  for (int num_test = 0; num_test < LABEL_LEN; num_test++)
  {
    for (int mm = 0; mm < 28; mm++)
      for (int nn = 0; nn < 28; nn++)
      {
        image1[28 * mm + nn] = (image[28 * 28 * num_test + 28 * mm + nn] > 0) ? 1 : ((image[28 * 28 * num_test + 28 * mm + nn] < 0) ? -1 : 0);
      }

    conv1(image1, w_conv11, b_conv11, o_conv1);
    avgpooling1(o_conv1, o_avgpooling1);
    conv2(o_avgpooling1, w_conv22, b_conv22, o_conv2);
    avgpooling2(o_conv2, o_avgpooling2);
    fc1(o_avgpooling2, w_fc11, b_fc11, o_fc1);
    fc2(o_fc1, w_fc22, b_fc22, o_fc2);
    fc3(o_fc2, w_fc33, b_fc33, o_fc3);
    softmax(o_fc3, probs);
    getmax(probs, result, num_test);
  }
}
