
# in evaluation

def simple_jieba(text):
    stop = StopWords()

    termlist = []
    seglist = jieba.cut(text)
    for term in seglist:
        term = term.lower()
        if len(term) > 1 and term not in stop.stopwords:
            termlist.append(term)
    return termlist


def normalize(arr):
    mean = np.mean(arr)
    std = np.std(arr)

    arr = arr - mean
    arr = arr / std

    return arr




class news():
    def tt_cooccur_matrix(self, td_matrix):
        last = runtime(last=None)
        vocab_size = td_matrix.shape[1]
        tt_cooccur_matrix = np.zeros((vocab_size, vocab_size))

        for d in range(self.data_config['train_size']):
            doc = td_matrix[d, :]
            if int(np.sum(doc)) != 0:
                wordid = np.nonzero(doc)[0].tolist()
                for i in range(len(wordid)):
                    if i == len(wordid) - 1:
                        continue
                    else:
                        for w2 in wordid[i+1:]:
                            tt_cooccur_matrix[wordid[i], w2] += 1
                            tt_cooccur_matrix[w2, wordid[i]] += 1

        last = runtime(last)
        np.save(self.data_config['path'] + 'tt_cooccur_matrix' + self.data_config['version'], tt_cooccur_matrix)    # tt_cooccur_matrix ((word for query) w1, w2 (nonzero if co-occur))

    def get_p_t2t1(self, tt_cooccur_matrix):
        vocab_size = tt_cooccur_matrix.shape[0]

        p_t2t1 = np.empty_like(tt_cooccur_matrix, dtype=float)

        for w in range(vocab_size):
            p_t2t1[w, :] = tt_cooccur_matrix[w, :]/np.sum(tt_cooccur_matrix[w, :])

        np.save(self.data_config['path'] + 'p_t2t1' + self.data_config['version'], p_t2t1)  # p_t2t1 ( t1, p(t2 | t1) )




    def find_word_with_cooccur(self,word, n_target):
        with open(self.data_config['path'] + 'corpus_info' + self.data_config['version'] + '.json', 'r', encoding='utf8') as f:
            corpus_info = json.load(f)

        tt_cooccur_matrix = np.load(self.data_config['path'] + 'tt_cooccur_matrix' + self.data_config['version'] + '.npy')

        word2id = corpus_info['word2id']
        id2word = corpus_info['id2word']

        cooccur_word = []
        if type(word) is str:
            if word in word2id.keys():
                # cooccur_word.append([word])
                wordid = word2id[word]
                tt_cooccur = tt_cooccur_matrix[wordid,:]
                top_wordid = np.argsort(tt_cooccur)[-1 * n_target:][::-1]
                for i in list(top_wordid):
                    cooccur_word.append(id2word[str(i)])
                print(cooccur_word)


        if type(word) is int:
            wordid = word
            # cooccur_word.append([id2word[str(word)]])
            tt_cooccur = tt_cooccur_matrix[wordid, :]
            top_wordid = np.argsort(tt_cooccur)[-1 * n_target:][::-1]
            for i in list(top_wordid):
                cooccur_word.append(id2word[str(i)])
            print(cooccur_word)

        if type(word) is list:
            for w in word:
                if w in word2id.keys():
                    # cooccur_word.append([word])
                    wordid = word2id[w]
                    tt_cooccur = tt_cooccur_matrix[wordid,:]
                    top_wordid = np.argsort(tt_cooccur)[-1 * n_target:][::-1]
                    for i in list(top_wordid):
                        cooccur_word.append(id2word[str(i)])
            cooccur_word = list(set(cooccur_word))

        return cooccur_word

    def build_QA(self, input_word, blacklist, n_target, category, add=False):
        cooccur_word = self.find_word_with_cooccur(input_word, n_target)

        with open(data_config['path'] + 'seed_word' + data_config['version'] + '.json', 'r', encoding='utf8') as f:
            seed_word = json.load(f)

        seed_wordlist = seed_word[category]  # '娛樂','運動', '國際','社會', '科技', '政治','財金'

        seed_wordlist += input_word

        new_word = list(set(cooccur_word) - set(seed_wordlist))
        new_word = list(set(new_word) - set(blacklist))

        print(new_word)

        if add is True:
            seed_word[category] = seed_wordlist

            print(category + 'wordlist length: ' + str(len(set(seed_wordlist))))

            with open(data_config['path'] + 'seed_word' + data_config['version'] + '.json', 'w', encoding='utf8') as f:
                json.dump(seed_word, f)



class Test_model():


    def _doc_vectorizer(self, text):
        word2id = self.corpus_info['word2id']
        p_zw = np.array(self.model['opt_p_zw'])
        z = p_zw.shape[0]

        if type(text) is str:
            doc_vec = np.zeros((z, 1))
            termlist = simple_jieba(text)
            for term in termlist:
                if term in word2id.keys():
                    wordid = word2id[term]
                    doc_vec += p_zw[:, wordid]

            doc_vec = doc_vec/len(termlist)

            doc_vec = doc_vec.T

            return doc_vec  #shape (1, z)

        if type(text) is list:
            doc_vec_C = np.zeros((z, 1))
            for doc in text:
                doc_vec = np.zeros((z, 1))
                termlist = simple_jieba(doc)
                for term in termlist:
                    if term in word2id.keys():
                        wordid = word2id[term]
                        doc_vec += p_zw[:, wordid]
                doc_vec = doc_vec / len(termlist)
                np.append(doc_vec_C, doc_vec, axis=1)   #doc_vec_C.shape (z, d)

            doc_vec_C = doc_vec_C.T

            return doc_vec_C  #doc_vec_C.shape (d, z)


    def doc_find_doc(self, inputdoc, doc_basket, n_find):
        input_vec = self._doc_vectorizer(inputdoc)
        basket_vec = self._doc_vectorizer(doc_basket)

        sim_matrix = cosine_similarity(input_vec, basket_vec)   #shape(#inputdoc , #doc_basket)   (1, #doc basket)

        top_docid = np.argsort(sim_matrix)[-1 * n_find:][::-1]

        for docid in top_docid:
            print(doc_basket[docid])

    def doc_weight_distr(self, opt=False, iters=False):
        td_matrix = self.td_matrix

        if opt is True:
            p_zw = np.array(self.model['opt_p_zw'])

            doc_hhi_collect = []
            doc_nonzero_collect = []
            for d in range(self.data_config['train_size']):
                doc_weight = td_matrix[d, :]* p_zw

                if int(np.sum(td_matrix[d, :])) != 0:
                    doc_weight = np.sum(doc_weight, axis=1)/int(np.sum(td_matrix[d, :]))   #shape(z,d=1)

                    hhi = get_hhi(doc_weight)
                    nonzero = np.count_nonzero(doc_weight)
                    doc_hhi_collect.append(hhi)
                    doc_nonzero_collect.append(nonzero)
                else:
                    print(' ')
            print('------- optimal')
            print('--- doc in # of cluster')
            print(stats.describe(np.array(doc_nonzero_collect)))
            print('--- doc distr hhi')
            print(stats.describe(np.array(doc_hhi_collect)))

        if iters is True:
            p_zw_in_iters = self.model['p_zw_in_iters']
            savepoint = list(p_zw_in_iters.keys())

            mean_hhi = []
            var_hhi = []
            for iter in savepoint:
                p_zw = np.array(p_zw_in_iters[iter])
                doc_hhi_collect = []
                doc_nonzero_collect = []
                for d in range(self.data_config['train_size']):
                    doc_weight = td_matrix[d, :] * p_zw

                    if int(np.sum(td_matrix[d, :])) != 0:
                        doc_weight = np.sum(doc_weight, axis=1)/int(np.sum(td_matrix[d, :]))   #shape(z,d=1)

                        hhi = get_hhi(doc_weight)
                        nonzero = np.count_nonzero(doc_weight)
                        doc_hhi_collect.append(hhi)
                        doc_nonzero_collect.append(nonzero)
                    else:
                        print(' ')
                print('------- in iter '+str(iter))
                print('--- doc in # of cluster')
                print(stats.describe(np.array(doc_nonzero_collect)))
                print('--- doc distr hhi')
                print(stats.describe(np.array(doc_hhi_collect)))

                mean_hhi.append(float(np.mean(doc_hhi_collect)))
                var_hhi.append(float(np.var(doc_hhi_collect)))
                print(mean_hhi)

            print(mean_hhi)
            print(var_hhi)

            x = []
            for point in savepoint:
                x.append(int(point))
            plt.ylim(0.3, 0.8)
            plt.plot(x, mean_hhi)
            plt.show()


    def word_weight_distr(self, opt=False, iters=False):

        if opt is True:
            word_hhi_collect = []
            word_nonzero_collect = []
            p_zw = np.array(self.model['opt_p_zw'])
            for w in range(self.corpus_info['vocab_size']):
                word_weight = p_zw[:, w]
                hhi = get_hhi(word_weight)
                nonzero = np.count_nonzero(word_weight)
                word_hhi_collect.append(hhi)
                word_nonzero_collect.append(nonzero)
            print('-------- optimal')
            print('--- word in # of cluster')
            print(stats.describe(np.array(word_nonzero_collect)))
            print('--- word distr hhi')
            print(stats.describe(np.array(word_hhi_collect)))

        if iters is True:
            p_zw_in_iters = self.model['p_zw_in_iters']
            savepoint = list(p_zw_in_iters.keys())

            for iter in savepoint:
                word_hhi_collect = []
                word_nonzero_collect = []
                p_zw = p_zw_in_iters[iter]
                for w in range(self.corpus_info['vocab_size']):
                    word_weight = p_zw[:, w]
                    hhi = get_hhi(word_weight)
                    nonzero = np.count_nonzero(word_weight)
                    word_hhi_collect.append(hhi)
                    word_nonzero_collect.append(nonzero)
                print('--------- in iter '+str(iter))
                print('--- word in # of cluster')
                print(stats.describe(np.array(word_nonzero_collect)))
                print('--- word distr hhi')
                print(stats.describe(np.array(word_hhi_collect)))


class Observe_Model():
    def __init__(self, data_config, train_plan, load_td_matrix=False):
        with open(data_config['path'] + 'lda_model'+ train_plan['version']+ '.json', 'r', encoding='utf8') as f:
            model = json.load(f)
        with open(data_config['path'] + 'corpus_info'+ data_config['version']+ '.json', 'r', encoding='utf8') as f:
            corpus_info = json.load(f)

        if load_td_matrix is True:
            td_matrix = np.load(data_config['path'] + 'td_matrix' + data_config['version']+'.npy')
            self.td_matrix = td_matrix

        self.train_plan = train_plan
        self.data_config = data_config

        self.corpus_info = corpus_info
        self.model = model


        print('---- maxlike')
        print(train_plan)
        print(model['maximum_likelihood'])
        print('')

    def maxlike_graph(self):
        likelihood_in_iters = self.model['likelihood_in_iters']

        likelihood_in_iters = {int(k): v for k, v in likelihood_in_iters.items()}

        iter = list(likelihood_in_iters.keys())
        likelihood = list(likelihood_in_iters.values())

        print(self.train_plan)
        plt.plot(iter, likelihood)
        plt.xlabel('iteration')
        plt.ylabel('loglikelihood')
        plt.show()

    def word_topic_distr(self):
        nzw = np.array(self.model['opt_nzw'])
        vocab_size = nzw.shape[1]

        kurt_list = []
        for i in range(vocab_size):
            nz1 = nzw[:, i]
            word_distr = nz1 / np.sum(nz1)
            kurt = stats.kurtosis(word_distr)
            kurt_list.append(kurt)

        print('---- word_topic_distribution')
        print(self.train_plan)
        print(stats.describe(kurt_list))
        print('')

    def doc_topic_distr(self):
        nmz = np.array(self.model['opt_nmz'])
        doc_size = nmz.shape[0]

        kurt_list = []
        for i in range(doc_size):
            n1z = nmz[i, :]
            if int(np.sum(n1z)) != 0:
                doc_distr = n1z / np.sum(n1z)
                kurt = stats.kurtosis(doc_distr)
                kurt_list.append(kurt)

            print('---- doc_topic_distribution')
            print(self.train_plan)
            print(stats.describe(kurt_list))
