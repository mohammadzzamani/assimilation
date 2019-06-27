from random import random

import pandas as pd
import numpy as np
from functools import partial
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from happierfuntokenizing.happierfuntokenizing import Tokenizer
tokenizer = Tokenizer()
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
root_dir='/Users/Mz/Downloads/assimilation/'
results_sub_dir = 'results_0619/'
big5_filename = 'big5.csv'
message_filename='message_topics.csv'
# message_stats_filename='message_stats.csv'
message_stats_filename='message_stats_all.csv'
stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves',
              'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
              'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
              'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
              'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about',
              'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up',
              'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
              'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
              'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now']

contagious_type = ['inf', 'match']
network = ['0', '1', '6']
cont = ['csim_lf', 'csim_lr', 'csim_fr', 'dot_lf', 'dot_lr', 'dot_fr']
big5 = ['ope', 'ext','neu', 'agr', 'con']


def read_data():

    ##### read contagious scores
    # contagious = ['match']
    # network = ['1', '6']
    prefix = 'to2001_'
    suffix = '_p2_cs.csv'
    data = None
    for c in contagious_type:
        for n in network:
            filename=prefix+ c + '_'+ n + suffix
            filename=prefix+ n+'_'+c + suffix
            try:
                d = pd.read_csv(root_dir+results_sub_dir+filename)
                d.set_index('topic', inplace=True)
                d.columns = [ col+'_'+c+''+n for col in d.columns]
                data = pd.merge(data, d, left_index=True, right_index=True, how='inner') if data is not None else d
                # print data.shape
                # print data.columns
            except:
                print 'no such file existed: {0}'.format(filename)


    '''
    inf_filenames = [ 'to2001_n1_f1_p2_cs.csv', 'to2001_n6_f1_p2_cs.csv', 'to2001_n0_f1_p2_cs.csv' ]
    match_filenames = [ 'to2001_n1_f2_p2_cs.csv', 'to2001_n6_f2_p2_cs.csv', 'to2001_n0_f2_p2_cs.csv']
    dfs = {'inf':[], 'match':[]}
    for filename in inf_filenames:
        try:
            dfs['inf'].append(pd.read_csv(root_dir+results_sub_dir+filename) )
        except:
            print 'no such file existed: {0}'.format(filename)
    for filename in match_filenames:
        try:
            dfs['match'].append(pd.read_csv(root_dir+results_sub_dir+filename) )
        except:
            print 'no such file existed: {0}'.format(filename)
    '''


    ##### read personality scores
    try:
        big5_data = pd.read_csv(root_dir+big5_filename)
        big5_data.set_index('topic', inplace=True)
        #print big5_data.shape
    except:
        print 'no such file({0}) existed'.format(big5_filename)

    data = pd.merge(data, big5_data, left_index=True, right_index=True, how='inner')
    #print data.columns
    #print data.shape

    return data

def get_tokens(desc):
    return [ word for word in tokenizer.tokenize( desc ) if word not in stopwords ]

def read_message_data():
    topics = pd.read_csv(root_dir+results_sub_dir+message_filename)
    stats = pd.read_csv(root_dir+results_sub_dir+message_stats_filename)
    topics_cp_filename = 'topics_cp.csv'
    topics_cp = pd.read_csv(root_dir+results_sub_dir+topics_cp_filename)
    stats['words'] = stats['message'].apply( partial( get_tokens))
    # print topics.columns
    print 'shapes: '
    print topics.shape
    # print stats.columns
    print stats.shape
    return topics, stats, topics_cp

def calc_contagious_score(data):
    print 'calc_contagious_score...'
    def func_calc_score(row, cols):
        suffix = '_inf0'
        [ angle_fr, angle_lf, angle_lr]  = [ np.arccos(row[csim+suffix])/np.pi for csim in ['csim_fr', 'csim_lf', 'csim_lr'] ]
        score =  angle_lr / angle_lf
        # score = row['csim_lf'+suffix]
        score = row['csim_lf'+suffix] - row['csim_lr'+suffix]
        # if score >= 0.1499:
        #     print 'row: ', row
        return score
    columns = list(data.columns.values)
    # data['score'] = data.apply(lambda x: random())
    data['score'] = data.apply(partial(func_calc_score, cols=columns) , axis=1)
    data['score_rank'] = data['score'].rank(ascending=False)
    data.sort_values(by=['score'], inplace=True, ascending=False)
    print data.score[:10]
    print data.score[-10:]
    return data

def calc_big5_corr(data):
    for c in big5:
        data[c+'_rank_'] = data[c].abs().rank(ascending=False)
    # print data[['con','con_rank', 'con_rank_'] ]
    data['avg_rank_big5'] = data.apply(lambda x: (np.average([x[c+'_rank_'] for c in big5])), axis=1)
    # print data[:10]
    data = data[ [c+'_rank' for c in big5 ] + ['avg_rank_big5']+ ['score_rank' ] ]
    print 'correlation of big5 score and contagious score'
    print data.corr()['score_rank']

    data.sort_values(by=['avg_rank_big5'], inplace=True)
    print 'average of contagious score rank for top 100 average of rank of abs of big5 score: ' , data[:100]['score_rank'].mean()
    print 'average of contagious score rank for bottom 100 average of rank of abs of big5 score: ' ,data[-100:]['score_rank'].mean()

def calc_message_score(topics, data, stats):
    print 'calc_message_score...'
    print stats.columns

    data.reset_index(inplace=True)

    print 'topics.columns: ', topics.columns
    print 'topics.shape: ', topics.shape
    data.reset_index(inplace=True)
    all_df = pd.merge(topics, data, left_on=['feat'], right_on=['topic'], how='inner')
    print 'all_df.columns: ', all_df.columns
    print 'all_df.shape: ', all_df.shape
    all_df = pd.merge(all_df, stats, left_on=['group_id'], right_on=['message_id'], how='inner')
    print 'all_df.columns: ', all_df.columns
    print 'all_df.shape: ', all_df.shape



    #data.reset_index(inplace=True)
    #topics = pd.merge(topics, data, left_on=['feat'], right_on=['topic'], how='inner')
    all_df['score_gn'] = all_df['score'] * all_df['group_norm']
    #print topics[:10][['id', 'group_id', 'score', 'score_gn', 'group_norm']]
    msg_df = all_df.groupby(['group_id', 'user_id','retweet_count' , 'favorite_count'])[['score_gn']].sum()
    msg_df.columns = ['msg_score_sum']
    #print sum_[:5]
    msg_df['msg_score_mean']= all_df.groupby(['group_id', 'user_id', 'retweet_count' , 'favorite_count'])[['score_gn']].mean()
    #print sum_[:5]
    # msg_df['count']= all_df.groupby(['group_id', 'user_id'])[['score_gn']].count()
    # #print sum_[:5]
    msg_df['msg_score_max']= all_df.groupby(['group_id', 'user_id', 'retweet_count' , 'favorite_count'])[['score_gn']].max()
    # #print sum_[:5]
    # msg_df['min']= all_df.groupby(['group_id', 'user_id'])[['score_gn']].min()
    msg_df.reset_index(inplace=True)
    print msg_df.columns
    #msg_df.columns = ['group_id' ,'user_id', 'msg_score_sum' , 'msg_score_mean'] #, 'msg_score_count', 'msg_score_max' , 'msg_score_min']

    print msg_df.columns
    print 'msg_df: '
    print msg_df[:5]


    user_df= stats.groupby(['user_id'])[['retweet_count']].mean()
    user_df['favorite'] = stats.groupby(['user_id'])[['favorite_count']].mean()
    user_df['count'] = stats.groupby(['user_id'])[['favorite_count']].count()
    user_df.columns = ['user_retweet', 'user_favorite', 'user_count']
    print ('user_df:')
    print user_df[:5]

    # temp = msg_df.groupby(['group_id', 'user_id'])[['score_gn']].sum()
    user_df['user_score'] = msg_df.groupby(['user_id'])[['msg_score_max']].mean()
    user_df.reset_index(inplace=True)

    print ">>>>>"
    print user_df.columns
    print msg_df.columns
    print user_df.shape
    print msg_df.shape
    print ">>>>>"
    print user_df[:10]
    print msg_df[:10]



    msg_df  = pd.merge(msg_df, user_df, left_on=['user_id'], right_on=['user_id'], how='inner')
    print msg_df.columns
    print msg_df.shape
    print msg_df[[ 'msg_score_sum', 'user_score']][:10]
    #print sum_[:5]
    msg_df['msg_score_dif'] = msg_df['msg_score_max'] - msg_df['user_score']
    msg_df['retweet_dif'] = msg_df['retweet_count'] - msg_df['user_retweet']
    msg_df['favorite_dif'] = msg_df['favorite_count'] - msg_df['user_favorite']
    print "-------"
    print msg_df[[ 'msg_score_dif', 'msg_score_max', 'user_score']][:10]
    print "-------"
    print msg_df.corr(method='spearman')[[ 'msg_score_dif', 'msg_score_max', 'user_score']]
    print "=========="
    # p1 = sum_.loc[sum_['user_id'] == 10870512]
    # print p1.shape
    # print p1.corr(method='spearman')[[ 'score_gn_dif']]
    # print p1[[ 'user_id',  'score_gn_x', 'score_gn_y', 'score_gn_dif', 'retweet_count', 'favorite_count' ]][:20]
    # p1 = sum_.loc[sum_['user_id'] == 12369372]
    # print p1.shape
    # print p1.corr(method='spearman')[['score_gn_dif']]
    # print p1[[ 'user_id',  'score_gn_x', 'score_gn_y', 'score_gn_dif', 'retweet_count', 'favorite_count' ]][:20]
    # p1 = sum_.loc[sum_['user_id'] == 13884162]
    # print p1.shape
    # print p1.corr(method='spearman')[['score_gn_dif']]
    # print p1[[ 'user_id',  'score_gn_x', 'score_gn_y', 'score_gn_dif', 'retweet_count', 'favorite_count' ]][:20]

    users = list(user_df.user_id.unique())
    scores = []
    for u in users:
        person = msg_df.loc[msg_df['user_id'] == u]
        #print type( p1.corr(method='spearman') )
        #print p1.corr(method='spearman')[['score_gn_dif']]
        # scores.append(person.corr(method='spearman')['retweet_count'][['msg_score_dif']].values )
        X = person[['group_id' , 'msg_score_max', 'user_score',  'user_retweet' ]]
        Y = person[['retweet_count', 'favorite_count']]
        scores.append(learn(X.values, Y.values))
    print scores
    print 'mean: ' , np.mean(scores)

    X = msg_df[['group_id' , 'msg_score_dif', 'user_score', 'user_retweet', 'user_favorite' ]]
    Y = msg_df[['retweet_count', 'favorite_count']]
    return X, Y


global_counter = 0

def calc_message_score2(topics_cp, data, stats):
    print 'calc_message_score2...'
    def message_score(message, words_score):
        try:
            vals = []
            val = 0
            count = 0
            for word in message:
                # val += words_score.loc[(words_score['term'] == word)]['word_score']
                # print words_score.loc[word]
                # print word in words_score.index
                if word in words_score.index:
                    vals.append(words_score.loc[word].values[0])
                    val += words_score.loc[word].values[0]
                    count+=1
        except:
            print '-----error-----'
            print type(message)
        vals = sorted(vals)
        val = np.sum(vals[-4:])
        count = min(2, max(count, 1))
        # return val  *1.0/count
        if len(vals) == 0:
            global global_counter
            global_counter +=1
            if global_counter % 100 == 1:
                print global_counter

        return vals[-5] if len(vals) > 5 else vals[-len(vals)] if len(vals) > 0 else 0
    topics_cp = pd.merge(topics_cp, data, left_on=['category'], right_on=['topic'], how='inner')
    topics_cp['word_score'] =  topics_cp['score'] * topics_cp['weight']
    # topics_1751= topics_cp[topics_cp['category']==1751]
    print topics_cp.columns
    topics_cp.sort_values(['term'], inplace=True)
    print  topics_cp[['id', 'term', 'category', 'word_score', 'weight', 'score']][:10]
    top10_scores = topics_cp.sort_values(['term','weight'],ascending=False).groupby('term').head(2)
    # print "______"
    # print top10_scores[:10]


    #df.groupby(['Mt'])['count'].transform(max) == df['count']

    print top10_scores.shape
    print top10_scores[:20][['term','word_score']]
    words_score = top10_scores.groupby(['term'])[['word_score']].mean()
    words_score.sort_values(['word_score'], ascending=False, inplace=True)
    print words_score[:10]
    print words_score[-10:]
    # words_score.reset_index(inplace=True)
    print '====='
    # print len(list(topics_cp.term.unique()))
    print words_score.shape
    print stats.shape
    stats['message_score'] = stats['words'].apply(partial(message_score, words_score=words_score))

    # print 'stats:'
    # print stats[:10]

    print stats.columns
    print topics_cp.columns
    print data.columns
    # return



    # data.reset_index(inplace=True)
    #
    # print 'topics.columns: ', topics.columns
    # print 'topics.shape: ', topics.shape
    # data.reset_index(inplace=True)
    # # all_df = pd.merge(topics, data, left_on=['feat'], right_on=['topic'], how='inner')
    # # print 'all_df.columns: ', all_df.columns
    # # print 'all_df.shape: ', all_df.shape
    # all_df = pd.merge(data, stats, left_on=['group_id'], right_on=['message_id'], how='inner')
    # print 'all_df.columns: ', all_df.columns
    # print 'all_df.shape: ', all_df.shape



    #data.reset_index(inplace=True)
    # #topics = pd.merge(topics, data, left_on=['feat'], right_on=['topic'], how='inner')
    # all_df['score_gn'] = all_df['score'] * all_df['group_norm']
    #print topics[:10][['id', 'group_id', 'score', 'score_gn', 'group_norm']]
    # msg_df = stats.groupby(['message_id', 'user_id','retweet_count' , 'favorite_count'])[['message_score']].sum()
    # msg_df.columns = ['msg_score_sum']
    # #print sum_[:5]
    # msg_df['msg_score_mean']= all_df.groupby(['group_id', 'user_id', 'retweet_count' , 'favorite_count'])[['score_gn']].mean()
    # #print sum_[:5]
    # # msg_df['count']= all_df.groupby(['group_id', 'user_id'])[['score_gn']].count()
    # # #print sum_[:5]
    # msg_df['msg_score_max']= all_df.groupby(['group_id', 'user_id', 'retweet_count' , 'favorite_count'])[['score_gn']].max()
    # # #print sum_[:5]
    # # msg_df['min']= all_df.groupby(['group_id', 'user_id'])[['score_gn']].min()
    # msg_df.reset_index(inplace=True)
    # print msg_df.columns
    #msg_df.columns = ['group_id' ,'user_id', 'msg_score_sum' , 'msg_score_mean'] #, 'msg_score_count', 'msg_score_max' , 'msg_score_min']

    # print msg_df.columns
    # print 'msg_df: '
    # print msg_df[:5]


    user_df= stats.groupby(['user_id'])[['retweet_count']].mean()
    user_df['favorite'] = stats.groupby(['user_id'])[['favorite_count']].mean()
    user_df['count'] = stats.groupby(['user_id'])[['favorite_count']].count()
    user_df.columns = ['user_retweet', 'user_favorite', 'user_count']
    print ('user_df:')
    print user_df[:5]


    print stats[:10]
    # temp = msg_df.groupby(['group_id', 'user_id'])[['score_gn']].sum()
    user_df['user_score'] = stats.groupby(['user_id'])[['message_score']].mean()
    user_df.reset_index(inplace=True)

    print ">>>>>"
    print user_df.columns
    print stats.columns
    print user_df.shape
    print stats.shape
    print ">>>>>"
    # print user_df[:10]
    # print msg_df[:10]


    stats['word_count'] = stats.words.apply(lambda x: len(x))
    msg_df  = pd.merge(stats, user_df, left_on=['user_id'], right_on=['user_id'], how='inner')
    msg_df['fav_ret'] = msg_df['favorite_count'] + msg_df['retweet_count']
    print msg_df.columns
    print msg_df.shape
    print msg_df[[ 'message_score', 'user_score']][:10]
    #print sum_[:5]
    msg_df['message_score_dif'] = msg_df['message_score'] - msg_df['user_score']
    msg_df['retweet_dif'] = msg_df['retweet_count'] - msg_df['user_retweet']
    msg_df['favorite_dif'] = msg_df['favorite_count'] - msg_df['user_favorite']
    print "-------"
    print msg_df[[ 'message_score_dif', 'message_score', 'user_score', 'word_count']][:10]
    print "-------"
    print msg_df.corr(method='spearman')[[ 'message_score_dif', 'message_score', 'user_score', 'word_count']]
    print "=========="

    # p1 = sum_.loc[sum_['user_id'] == 10870512]
    # print p1.shape
    # print p1.corr(method='spearman')[[ 'score_gn_dif']]
    # print p1[[ 'user_id',  'score_gn_x', 'score_gn_y', 'score_gn_dif', 'retweet_count', 'favorite_count' ]][:20]
    # p1 = sum_.loc[sum_['user_id'] == 12369372]
    # print p1.shape
    # print p1.corr(method='spearman')[['score_gn_dif']]
    # print p1[[ 'user_id',  'score_gn_x', 'score_gn_y', 'score_gn_dif', 'retweet_count', 'favorite_count' ]][:20]
    # p1 = sum_.loc[sum_['user_id'] == 13884162]
    # print p1.shape
    # print p1.corr(method='spearman')[['score_gn_dif']]
    # print p1[[ 'user_id',  'score_gn_x', 'score_gn_y', 'score_gn_dif', 'retweet_count', 'favorite_count' ]][:20]

    users = list(user_df.user_id.unique())
    scores = []
    errors = []
    allData = None #pd.DataFrame(columns = ['message_score', 'word_count', 'retweet_count', 'favorite_count', 'fav_ret'])
    for u in users:

        # print 'user: ' , u
        person = msg_df.loc[msg_df['user_id'] == u]
        print 'person: ',
        print person[:1]
        person[['message_score_norm', 'word_count_norm', 'retweet_count_norm', 'favorite_count_norm', 'fav_ret_norm']] = person[['message_score', 'word_count', 'retweet_count', 'favorite_count', 'fav_ret']].transform(lambda x: (x - x.min()) / (x.max() - x.min()))
        allData = allData.append(person) if allData is not None else person
        # print person[:5]
        #print type( p1.corr(method='spearman') )
        #print p1.corr(method='spearman')[['score_gn_dif']]
        # scores.append(person.corr(method='spearman')['retweet_count'][['msg_score_dif']].values )
        XNormed = person[[ 'message_score_norm', 'word_count_norm' ]]
        YNormed = person[['retweet_count_norm', 'favorite_count_norm', 'fav_ret_norm']]
        # print X[:10]
        # print Y[:10]
        # XNormed = X.transform(lambda x: (x - x.mean()) / x.std())
        # YNormed = Y.transform(lambda x: (x - x.mean()) / x.std())
        #XNormed = X.transform(lambda x: (x - x.min()) / (x.max() - x.min()))
        #YNormed = Y.transform(lambda x: (x - x.min()) / (x.max() - x.min()))
        # print XNormed[:10]
        # print YNormed[:10]
        #norm_msg_df['user_id'] = msg_df['user_id']

        err = [learn(XNormed.values,YNormed['retweet_count_norm'].values), learn(XNormed.values,YNormed['favorite_count_norm'].values),
               learn(XNormed.values,YNormed['fav_ret_norm'].values)]
        print 'errors: ', err
        errors.append(err)

        cor = [ person.corr(method='spearman')['retweet_count_norm'][['message_score_norm']].values[0],
                person.corr(method='spearman')['favorite_count_norm'][['message_score_norm']].values[0] ]

        scores.append( cor )
        # scores.append(learn(X.values, Y.values))
        # if person.corr(method='spearman')['favorite_dif'][['message_score_dif']].values[0] is :
        print u, ' ---- ' , cor

        # print u, ' ===== ', learn(X,Y[['retweet_count']].values), learn(X,Y[['favorite_count']].values)
        # if u == 1000774783:
        #     print 'person: ', person
    allData = allData.drop(['message', 'words', 'message_score_dif',  'retweet_dif',  'favorite_dif'], axis=1)
    print 'allllllll: '
    print allData[:2]
    allData.to_csv(root_dir+results_sub_dir+'score.csv', index=False)
    print scores
    print 'mean: ' , np.mean(scores,0)
    print 'mean: ', np.mean(errors,0)

    X = msg_df[['message_id' , 'message_score_dif', 'user_score', 'user_retweet', 'user_favorite' ]]
    Y = msg_df[['retweet_count', 'favorite_count']]
    return X, Y


def learn(X, Y):
    print 'learn...'
    print X.shape, ' , ', Y.shape
    x = X
    y = Y
    # print x[:5]
    # print y[:5]
    # print x
    # print '<---->'
    # print y
    reg = LinearRegression().fit(x, y) #max_iter=100, tol=1e-4)
    reg.fit(x, y)
    ypred = reg.predict(x)
    rmse1 = np.sqrt(mean_squared_error(y, ypred))
    print 'rmse: ', rmse1  ,  '  coef: ' , reg.coef_
    # print 'coef: ', reg.coef_, ' , ', reg.intercept_

    # val = np.mean(y)
    # print 'val: ', val
    # ypred2 = [ val for i in y ]
    # print X.shape
    # print X
    x=X[:,1].reshape(-1,1)
    reg = LinearRegression().fit(x, y) #max_iter=100, tol=1e-4)
    reg.fit(x, y)
    ypred2 = reg.predict(x)
    rmse2 = np.sqrt(mean_squared_error(y, ypred2))
    print 'rmse: ', rmse2   ,  '  coef: ' , reg.coef_

    # x = X[:,3:]
    # y = Y[:,0]
    # reg = LinearRegression().fit(x, y) #max_iter=100, tol=1e-4)
    # reg.fit(x, y)
    # ypred = reg.predict(x)
    # rmse3 = np.sqrt(mean_squared_error(y, ypred))
    # print 'rmse: ', rmse3
    # # print 'coef: ', reg.coef_, ' , ', reg.intercept_
    print '----'

    return rmse1 - rmse2

data = read_data()
topics, stats, topics_cp = read_message_data()

data = calc_contagious_score(data)

calc_big5_corr(data)

X, Y = calc_message_score2(topics_cp, data, stats)

learn(X.values, Y.values)



