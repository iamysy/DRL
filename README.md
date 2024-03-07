# 替代模型的使用
# 使用之前训练过的数据进行LSTM训练
## 首先需要对数据进行处理
### 此内容用来记录遇到的问题

#### 首先需要确定我们输入和输出的构成，经过考虑后，决定将数据作为以下形式：

'''
 不含header
      输入                       输出
Time State Action | Next_Time_State Next_Time_Cd_Cl

'''
