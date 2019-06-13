# JSIS3D

This is the official Pytorch implementation of the following publication.

> **JSIS3D: Joint Semantic-Instance Segmentation of 3D Point Clouds with**<br/>
> **Multi-Task Pointwise Networks and Multi-Value Conditional Random Fields**<br/>
> Quang-Hieu Pham, Duc Thanh Nguyen, Binh-Son Hua, Gemma Roig, Sai-Kit
> Yeung<br/> *Conference on Computer Vision and Pattern Recognition (CVPR),
> 2019* (**Oral**)<br/>
> [Paper](https://pqhieu.github.io/assets/cvpr19/main.pdf),
> [Homepage](https://pqhieu.github.io/cvpr19.html)

### Citation
If you find our work useful for your research, please consider citing:

    @inproceedings{pham-jsis3d-cvpr19,
      title = {{JSIS3D}: Joint semantic-instance segmentation of 3d point clouds with multi-task pointwise networks and multi-value conditional random fields},
      author = {Pham, Quang-Hieu and Nguyen, Duc Thanh and Hua, Binh-Son and Roig, Gemma and Yeung, Sai-Kit},
      booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
      year = {2019}
    }

## Usage

### Prerequisites
This code is tested in Manjaro Linux with CUDA 10.0 and Pytorch 1.0.

- Python 3.5+
- Pytorch 0.4.0+

### Installation
To use MV-CRF (optional), you first need to compile the code:

    cd external/densecrf
    mkdir build
    cd build
    cmake -D CMAKE_BUILD_TYPE=Release ..
    make
    cd ../../.. # You should be at the root folder here
    make

### Dataset
We have preprocessed the S3DIS dataset ([2.5GB](https://drive.google.com/open?id=1s1cFfb8cInM-SNHQoTGxN9BIyNpNQK6x))
in HDF5 format. After downloading the files, put them into the corresponding
`data/s3dis/h5` folder.

### Training & Evaluation
To train a model on S3DIS dataset:

    python train.py --config configs/s3dis.json --logdir logs/s3dis

Log files and network parameters will be saved to the `logs/s3dis` folder.

After training, we can use the model to predict semantic-instance segmentation
labels as follows:

    python pred.py --logdir logs/s3dis --mvcrf

To evaluate the results, run the following command:

    python eval.py --logdir logs/s3dis

For more details, you can use the `--help` option for every scripts.

### Prepare your own dataset
Check out the `scripts` folder to see how we prepare the dataset for training.

## License
Our code is released under MIT license (see LICENSE for more details).

**Contact**: Quang-Hieu Pham (pqhieu1192@gmail.com)
