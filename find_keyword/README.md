# Find Keyword

  - text clustering 
  - 用 LDA collapsed gibbs sampling 算出 corpus 中每個字屬於第 n 個 topic 的機率
  - 亦可將該機率視為每個字的feature使用

# 主要功能
 - term_document_matrix.py
  讀取文本並將其轉換成 term_document matrix 
 
 - lda_colla_gibbs_model/sampler.py
 利用 term_document matrix 算出 LDA model 下的機率分佈 

 - word_predictor.py
 將 p( z | w) 視為每個字的feature，用 cosine similarity 找出與目標字相似的字

 - evaluate_text_model.py
 根據使用者自訂的 ground truth 檢驗 LDA model performance 

- lda_hyper_learn_operator/manager.py
 找出不同 datasets 下，最佳的 LDA model hyperparameter alpha, beta 


# 使用方法

main.py 為使用 options_parser 做成的執行檔
執行 flows.py 內的功能

    python3 main.py < flow >

#### flows

    -p , --prepare data 
從mongo撈文章切詞後，產生term_document matrix
 -  term_document matrix 以 numpy格式儲存
 - 其他關於corpus 的資訊，以json檔儲存 
 - 計算出p  (w ) ，以numpy格式儲存 
 - 字與id的對照表以dict的方式儲存為 word_id_converter
   word_id_converter{ word2id : {word: id} , id2word : { id : word } }

以上檔案皆存在 config['path'] 下

    -b , --build_and_evaluate
讀取 term_document matrix後，建立LDA model，並使用ground truth 進行 evaluation
###### PS : ground_truth.json 必須放在 config['path] 的資料夾下
.

    -a, --prepare_build_and_evaluate
   結合以上兩步

    -t , --tune_alpha_beta
 找出最佳的 alpha, beta
###### PS : 必須先prepare data
.

# 細節設定  
修改 main.py 檔案內的config即可對上述功能做細節調整
 - path 
   執行過程產生的各類檔案皆儲存在 path 下

- model_ver
  model version , 將會標住在model 相關的檔案檔名的後方

- train_size
 讀取幾篇文章

- max_df , min_df
    建立term document matrix時，將會以每個字在 corpus 內的document frequency (df)
    進行篩選，最後只會留下介於max_df , min_df之間的字
    ( 0 < df < 1 ,  df 為該字出現在data中 n%的文章內 )
- alpha, beta , n_topics 
    LDA hyperparameter
    在tune_alpha_beta 時為 initial parameter

- maxiter
    建立model時，進行sampling的次數

- step 
    tune_alpha_beta 時，alpha beta調整的步伐
