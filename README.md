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
读取info_list_i.csv与history_coef.csv文件中以上数据，并把他们拼接为一个excel表格后，需要把数据进一步处理
此处在处理state时遇到一个问题：state的形式很乱存入变量后，它的dtypes为object，存在多种数据类型，需要对其内部进行处理消除不需要的字符
'''
for i in range(0, 100):
 next_state = next_state.iloc[i]
 next_state = re.findall(r'',next_state)
 state.append(next_state)
'''
