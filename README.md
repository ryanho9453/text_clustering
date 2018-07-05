# Find Keyword
  - text clustering 
  - 用 LDA collapsed gibbs sampling 算出 corpus 中每個字屬於第 n 個 topic 的機率
  - 亦可將該機率視為每個字的feature使用

# 主要區塊
##### prepare_data/
  讀取文本並將其轉換成 term_document matrix 

require :
 -  (1) stopwords folder 
 - (2) whitelist.json , whitelist.txt ( 關鍵字list , 確保在切詞 建立term_document matrix時不被篩掉)
 
##### lda_collapsed_gibbs/model.py
 利用 term_document matrix 算出 LDA model 下的機率分佈 

require :
 - prior_cluster.json (事先決定好的關鍵字分群)


##### lda_collapsed_gibbs/word_predictor.py
 將 p( z | w) 視為每個字的feature，用 cosine similarity 找出與目標字相似的字

##### lda_collapsed_gibbs/evaluate_text_model.py
 根據使用者自訂的 ground truth 檢驗 LDA model performance 

require :
 - ground_truth.json ( 用來檢驗accuracy 的ground_truth )


##### parameter_learn/
 找出不同 datasets 下，最佳的 LDA model hyperparameter alpha, beta 


### 儲存路徑
package 將自動增加一個working directory 在 find_keyword下
以儲存package 中產生的檔案
working directory 的名字可以於config['path']中修改
