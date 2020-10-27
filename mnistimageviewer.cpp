#include "mnistimageviewer.h"

MnistImageViewer::MnistImageViewer(std::vector<double> image, QWidget *parent)
{
    mnistImage = new QGraphicsScene();
    mnistImage->setSceneRect(0,0,280,280);

    setFixedSize(280,280);
    setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    setScene(mnistImage);
    QBrush brush;
    brush.setStyle(Qt::SolidPattern);
    brush.setColor(Qt::gray);
    setBackgroundBrush(brush);

    for (int i = 0; i < 28; ++i)
    {
        for (int j = 0; j < 28; ++j)
        {
            QColor color{(int)image[i + 28 * j], (int)image[i + 28 * j], (int)image[i + 28 * j]};
            brush.setColor(color);
            QGraphicsRectItem* square = new QGraphicsRectItem;
            square->setRect(10*i,10*j,10,10);
            square->setBrush(brush);
            mnistImage->addItem(square);
            rectList.push_back(square);
        }
    }

}

MnistImageViewer::~MnistImageViewer()
{
    for (auto rect : rectList)
    {
        delete rect;
    }
}
