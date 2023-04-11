#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QFileDialog>
#include <QMessageBox>
#include <QPixmap>
#include <QImage>
#include <QVector>
#include <QQueue>
#include <QDebug>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include "image.hpp"
#include "sift.hpp"

using namespace cv;
using namespace std;

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
}

MainWindow::~MainWindow()
{
    delete ui;
}


void MainWindow::on_pushButton_clicked()
{
    QString imagePath;

    imagePath = QFileDialog::getOpenFileName(this, "Open an Image", "..", "Images (.png.xpm .jpg.bmb)");

    // read image using opencv
    if(imagePath.isEmpty())
        return;

    Mat image = imread(imagePath.toStdString());

//    std::ios_base::sync_with_stdio(false);
//    std::cin.tie(NULL);

//    if (argc != 2) {
//        std::cerr << "Usage: ./find_keypoints input.jpg (or .png)\n";
//        return 0;
//    }
//    Image image(argv[1]);
//    image =  image.channels == 1 ? image : rgb_to_grayscale(image);

//    std::vector<sift::Keypoint> kps = sift::find_keypoints_and_descriptors(image);
//    Image result = sift::draw_keypoints(image, kps);
//    result.save("result.jpg");

//    std::cout << "Found " << kps.size() << " keypoints. Output image is saved as result.jpg\n";
}

