#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QLabel>
#include <QKeyEvent>
#include <deque>
#include <opencv2/opencv.hpp>

struct CaptureParams {
    int width;
    int height;
    int fps;
};

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(const CaptureParams& p0,
                        const CaptureParams& p1,
                        QWidget* parent = nullptr);
    ~MainWindow();

protected:
    void keyPressEvent(QKeyEvent* event) override;

private:
    void openCams();
    void closeCams();
    void updateFrame();
    cv::Mat ensureGray(const cv::Mat& src);
    double computeSharp(const cv::Mat& gray);
    void drawOverlay(cv::Mat& img, const std::string& text);
    cv::Mat makeGraphImage(int w, int h);

    CaptureParams p0_, p1_;
    int CAP_W_, CAP_H_, DISP_W_, DISP_H_;

    cv::VideoCapture cam0_, cam1_;
    cv::Mat gaussMask_;
    QTimer* timer;
    QLabel* viewLabel;

    std::deque<std::pair<int, int>> history_;
    const int MAX_HISTORY_ = 100;
    bool freezeHistory_ = false;
    bool showGraph_ = false;

    enum class ViewMode { SideBySide, Analytics4Q };
    ViewMode mode_ = ViewMode::Analytics4Q;
};

#endif // MAINWINDOW_H
