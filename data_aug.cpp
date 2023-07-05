#include <iostream>
#include <random>
#include <opencv2/opencv.hpp>


using namespace cv;

bool SAVE_DEBUG = false;

bool colorChanges(Mat srcImg, Mat& dstImg)
{
    bool success = false;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist_gamma(0.5f, 1.5f); // Range for Gamma changes
    std::uniform_int_distribution<int> dist_brightness(-50, 50); // Range for Gamma changes

    dstImg = srcImg.clone();

    for (int i=0; i< srcImg.rows; i++)
    {
        for (int j=0; j< srcImg.cols; j++)
        {
            Vec3b& pixels = dstImg.at<Vec3b>(i, j); // get the pixel values
            // apply gamma for each channels
            for (int ch=0; ch < 3; ch++)
            {
                float gamma = dist_gamma(gen); // geenrate random values
                float intensity_g = pixels[ch] / 255.0f; // Normlize pixels
                intensity_g = std::pow(intensity_g, gamma) * 255.0f;

                // Apply threshold to limit extreme changes
                const float threshold = 0.05f;
                if (std::abs(intensity_g - pixels[ch]) > threshold * 255.0f)
                {
                    intensity_g = pixels[ch] + threshold * 255.0f;
                }

                int brightness = dist_brightness(gen);
                int intensity_b = pixels[ch] + brightness;

                intensity_g = std::max(0.0f, std::min(intensity_g, 255.0f));
                intensity_b = std::max(0, std::min(intensity_b, 255));


                pixels[ch] = static_cast<uchar>(intensity_g);
                pixels[ch] = static_cast<uchar>(intensity_b);
            }
        }
    }

    success = true;
    return success;
}

Mat aligned_img(const Mat& srcImg, const Mat& alteredImg)
{
    Mat alignedImg(srcImg.rows, srcImg.cols + alteredImg.cols, CV_8UC3);

    // Copy the source image to the left side
    for (int i = 0; i < srcImg.rows; i++)
    {
        for (int j = 0; j < srcImg.cols; j++)
        {
            alignedImg.at<Vec3b>(i, j) = srcImg.at<Vec3b>(i, j);
        }
    }

    // Copy the altered image to the right side
    for (int i = 0; i < alteredImg.rows; i++)
    {
        for (int j = 0; j < alteredImg.cols; j++)
        {
            alignedImg.at<Vec3b>(i, j + srcImg.cols) = alteredImg.at<Vec3b>(i, j);
        }
    }

    return alignedImg;
}



void usage()
{
    printf(" Resize images to 512 and make a aligined ddataset. ");
    printf(" -------------------------------------------------- ");
    printf(" ./data_aug srcImg Path  dstimg Path");
    printf(" -------------------------------------------------- ");
}

int main(int argc, char** argv) 
{
    if (argc < 2)
    {
        usage();
    }

    std::string imgPath = argv[1];
    std::string imgDst = argv[2];

    Mat srcImg = imread(imgPath);
    if (srcImg.empty())
    {
        printf("Failed to read the image.\n");
        return EXIT_FAILURE;
    }

    Mat resized_srcImg;

   resize(srcImg, resized_srcImg, Size(512, 512), INTER_LINEAR);

    Mat dstImg(512, 512, CV_8UC3);
    if (!colorChanges(resized_srcImg, dstImg))
    {
        printf("Could not apply img_colr function, give up.\n\n");
        return EXIT_FAILURE;
    }

    Mat alginedImg(Size(1024, 512), CV_8UC3);

    alginedImg = aligned_img(resized_srcImg, dstImg);

    if (!imwrite(imgDst, alginedImg))
    {
        printf("Could not save the image, give up.\n\n");
        return EXIT_FAILURE;
    }

    std::cout << "Augmentation is done. " << std::endl;  

    return EXIT_SUCCESS;
}
