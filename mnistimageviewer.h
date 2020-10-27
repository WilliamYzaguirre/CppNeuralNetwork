#ifndef MNISTIMAGEVIEWER_H
#define MNISTIMAGEVIEWER_H

#include <QObject>
#include <vector>
#include <QtWidgets/QGraphicsView>
#include <QGraphicsRectItem>


class MnistImageViewer : public QGraphicsView
{
    Q_OBJECT
public:
    MnistImageViewer(std::vector<double> image, QWidget *parent = 0);

    ~MnistImageViewer();

private:
    QGraphicsScene* mnistImage;
    std::vector<QGraphicsRectItem*> rectList;
};

#endif // MNISTIMAGEVIEWER_H
