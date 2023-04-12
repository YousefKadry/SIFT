#include "mainwindow.h"

#include <QApplication>
#include <QFile.h>


int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    // stylesheet
    QFile styleSheetFile("C:/Users/20115/Desktop/SIFT/Themes/ManjaroMix.qss");
    styleSheetFile.open(QFile::ReadOnly);
    QString styleSheet = QLatin1String(styleSheetFile.readAll());
    a.setStyleSheet(styleSheet);

    MainWindow w;
    w.show();
    return a.exec();
}


