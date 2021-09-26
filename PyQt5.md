# PyQt5

## 点击子窗口右上角的x后返回主窗口

```python
class Enter_raw_materials(QWidget, Ui_Enter_raw_materials):
    def __init__(self):
        super(Enter_raw_materials, self).__init__()
        self.setupUi(self)
    
    def closeEvent(self, event):# 重写closeevent事件
        self.close()
        main_window.show()
```

## 设置控件不可用

```python
self.lineEdit.setEnabled(False)
```

## 通知窗口

```python
from PyQt5.QtWidgets import QMessageBox
QMessageBox.warning(self, '错误', '开始时间不能晚于结束时间')
```

## tableWidget

```python
from PyQt5.QtWidgets import QWidget, QTableWidgetItem
```

- 设置表格窗口行数和列数

  ```python
  self.tableWidget.setColumnCount(3) # 列数
  self.tableWidget.setRowCount(3) # 行数
  ```

- 设置行名和列名

  ```python
  self.tableWidget.setVerticalHeaderLabels(['元素名称','元素缩写','元素含量']) # 行名
  self.tableWidget.setHorizontalHeaderLabels(['时间', '河道站点a', '河道站点b', '水库a', '水库b', '雨量站点a']) # 列名
  ```

- 设置行名列名是否可见

  ```python
  self.tableWidget.verticalHeader().setVisible(False) # 行名不可见
  self.tableWidget.horizontalHeader().setVisible(False) # 列名不可见
  ```

- 显示表格

  ```python
  self.tableWidget.setColumnCount(5)
  self.tableWidget.setRowCount(5)
  for i in range(0, 5):
      for j in range(0, 5):
          self.tableWidget_3.setItem(i, j, QTableWidgetItem(str(0)))
  ```

## MessageBox弹窗

- 带选项的弹窗

  ```python
  reply = QMessageBox.warning(self, '警告', '确认删除所选用户？', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
  ```

- 不带选项的弹窗

  ```python
   QMessageBox.warning(self, '错误', '请输入合法的元素名称')
  ```

## ComboBox

- 设置可选项目

  ```python
  self.comboBox.addItems([1,2,3])
  ```

- 获取选择的项目

  ```python
  self.comboBox.currentText()
  ```

## radiobutton

- 信号

  ```python
   self.radioButton.toggled.connect()
  ```

- 检查状态

  ```python
  self.radioButton.isChecked()
  ```

## tabWidget

- 信号

  ```python
  self.tabWidget.currentChanged.connect()
  ```

- 当前页面

  ```python
  self.tabWidget.currentIndex()
  ```

## label

- 设置label内文字

  ```python
  self.label_2.setText()
  ```

  

