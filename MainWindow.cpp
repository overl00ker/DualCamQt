#include "MainWindow.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <QMessageBox>
#include <QTimer>
#include <QVBoxLayout>
#include <QLabel>
#include <opencv2/opencv.hpp>

MainWindow::MainWindow(const CaptureParams& p0,
                       const CaptureParams& p1,
                       QWidget* parent)
    : QMainWindow(parent),
      p0_(p0), p1_(p1)
{
    std::cout << "MainWindow constructor started" << std::endl;

    CAP_W_ = p0_.width;
    CAP_H_ = p0_.height;
    DISP_W_ = CAP_W_ * 2;
    DISP_H_ = CAP_H_ * 2;

    QWidget* central = new QWidget(this);
    auto* layout = new QVBoxLayout(central);
    viewLabel = new QLabel;
    viewLabel->setFixedSize(DISP_W_, DISP_H_);
    layout->addWidget(viewLabel);
    setCentralWidget(central);
    setFixedSize(DISP_W_, DISP_H_);

    cv::Mat ky = cv::getGaussianKernel(CAP_H_, CAP_H_ / 6.0, CV_64F);
    cv::Mat kx = cv::getGaussianKernel(CAP_W_, CAP_W_ / 6.0, CV_64F);
    gaussMask_ = ky * kx.t();
    gaussMask_ /= cv::sum(gaussMask_)[0];

    openCams();

    std::cout << "cam0 opened: " << cam0_.isOpened() << std::endl;
    std::cout << "cam1 opened: " << cam1_.isOpened() << std::endl;

    if (!cam0_.isOpened() || !cam1_.isOpened()) {
        std::cerr << "\u274C Cameras failed to open. Exiting." << std::endl;
        QMessageBox::critical(this, "Camera Error", "Failed to open one or both cameras.");
        return;
    }

    timer = new QTimer(this);
    connect(timer, &QTimer::timeout, this, &MainWindow::updateFrame);
    timer->start(1000 / std::max(1, p0_.fps));

    std::cout << "MainWindow initialized successfully" << std::endl;
}

MainWindow::~MainWindow()
{
    closeCams();
}

void MainWindow::openCams()
{
    std::string gst0 = "libcamerasrc camera-name=/base/axi/pcie@1000120000/rp1/i2c@88000/imx296@1a ! video/x-raw,width=640,height=480,format=YUY2 ! videoconvert ! appsink";
    std::string gst1 = "libcamerasrc camera-name=/base/axi/pcie@1000120000/rp1/i2c@80000/imx296@1a ! video/x-raw,width=640,height=480,format=YUY2 ! videoconvert ! appsink";

    cam0_.open(gst0, cv::CAP_GSTREAMER);
    cam1_.open(gst1, cv::CAP_GSTREAMER);
}

void MainWindow::closeCams()
{
    cam0_.release();
    cam1_.release();
}

cv::Mat MainWindow::ensureGray(const cv::Mat& src)
{
    if (src.channels() == 1) return src;
    cv::Mat g;
    cv::cvtColor(src, g, cv::COLOR_BGR2GRAY);
    return g;
}

double MainWindow::computeSharp(const cv::Mat& gray)
{
    cv::Mat lap, absL, weighted;
    cv::Laplacian(gray, lap, CV_64F);
    cv::absdiff(lap, cv::Scalar::all(0), absL);
    cv::multiply(absL, gaussMask_, weighted);
    return cv::sum(weighted)[0];
}

void MainWindow::drawOverlay(cv::Mat& img, const std::string& text)
{
    cv::putText(img, text, cv::Point(5,20),
                cv::FONT_HERSHEY_SIMPLEX, 0.7,
                cv::Scalar(0,0,255), 2);
}

cv::Mat MainWindow::makeGraphImage(int w, int h)
{
    cv::Mat img(h, w, CV_8UC3, cv::Scalar(0,0,0));
    if (history_.size() < 2) return img;
    double dx = double(w) / (MAX_HISTORY_ - 1);
    for (int i = 1; i < (int)history_.size(); ++i)
    {
        int x0 = int((i-1)*dx), y0 = h - history_[i-1].first  * h / 100;
        int x1 = int(i*dx),     y1 = h - history_[i].first    * h / 100;
        cv::line(img, {x0,y0}, {x1,y1}, {0,255,0}, 2);
        int y0b = h - history_[i-1].second * h / 100;
        int y1b = h - history_[i].second   * h / 100;
        cv::line(img, {x0,y0b}, {x1,y1b}, {0,0,255}, 2);
    }
    return img;
}

void MainWindow::updateFrame()
{
    cv::Mat f0, f1;
    bool g0 = cam0_.grab();
    bool g1 = cam1_.grab();
    if (g0) cam0_.retrieve(f0); else cam0_.read(f0);
    if (g1) cam1_.retrieve(f1); else cam1_.read(f1);
    if (f0.empty() || f1.empty()) return;

    cv::Mat gray0 = ensureGray(f0);
    cv::Mat gray1 = ensureGray(f1);

    if (gray0.cols != CAP_W_ || gray0.rows != CAP_H_)
        cv::resize(gray0, gray0, cv::Size(CAP_W_, CAP_H_));
    if (gray1.cols != CAP_W_ || gray1.rows != CAP_H_)
        cv::resize(gray1, gray1, cv::Size(CAP_W_, CAP_H_));

    double s0 = computeSharp(gray0);
    double s1 = computeSharp(gray1);
    double maxSharp = std::max(s0, s1);
    int pct0 = maxSharp > 0 ? int(s0 / maxSharp * 100) : 0;
    int pct1 = maxSharp > 0 ? int(s1 / maxSharp * 100) : 0;

    if (!freezeHistory_)
    {
        if (history_.empty() ||
            std::abs(pct0 - history_.back().first)  >= 1 ||
            std::abs(pct1 - history_.back().second) >= 1)
        {
            history_.emplace_back(pct0, pct1);
            if ((int)history_.size() > MAX_HISTORY_) history_.pop_front();
        }
    }

    cv::Mat disp0, disp1;
    cv::cvtColor(gray0, disp0, cv::COLOR_GRAY2BGR);
    cv::cvtColor(gray1, disp1, cv::COLOR_GRAY2BGR);
    drawOverlay(disp0, std::to_string(pct0) + "%");
    drawOverlay(disp1, std::to_string(pct1) + "%");

    cv::Mat diff, diffC;
    cv::absdiff(gray0, gray1, diff);
    cv::cvtColor(diff, diffC, cv::COLOR_GRAY2BGR);

    cv::Mat graph = (showGraph_ ? makeGraphImage(CAP_W_, CAP_H_)
                                : cv::Mat::zeros(CAP_H_, CAP_W_, CV_8UC3));

    cv::Mat canvas;
    if (mode_ == ViewMode::Analytics4Q)
    {
        canvas = cv::Mat(DISP_H_, DISP_W_, CV_8UC3, cv::Scalar(0,0,0));
        disp0.copyTo(canvas(cv::Rect(0,          0,      CAP_W_, CAP_H_)));
        disp1.copyTo(canvas(cv::Rect(CAP_W_,     0,      CAP_W_, CAP_H_)));
        diffC.copyTo(canvas(cv::Rect(0,          CAP_H_, CAP_W_, CAP_H_)));
        graph.copyTo(canvas(cv::Rect(CAP_W_,     CAP_H_, CAP_W_, CAP_H_)));
    }
    else // SideBySide
    {
        canvas = cv::Mat(CAP_H_, CAP_W_*2, CV_8UC3);
        disp0.copyTo(canvas(cv::Rect(0,0,CAP_W_,CAP_H_)));
        disp1.copyTo(canvas(cv::Rect(CAP_W_,0,CAP_W_,CAP_H_)));
    }

    QImage img(canvas.data, canvas.cols, canvas.rows, int(canvas.step), QImage::Format_BGR888);
    viewLabel->setPixmap(QPixmap::fromImage(img));
}

void MainWindow::keyPressEvent(QKeyEvent* event)
{
    switch (event->key())
    {
    case Qt::Key_F: freezeHistory_ = !freezeHistory_; break;
    case Qt::Key_T: showGraph_    = !showGraph_;      break;
    case Qt::Key_S:
        mode_ = (mode_ == ViewMode::Analytics4Q ? ViewMode::SideBySide
                                                : ViewMode::Analytics4Q);
        break;
    case Qt::Key_Escape: close(); break;
    default: QMainWindow::keyPressEvent(event);
    }
}
