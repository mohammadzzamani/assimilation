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
message_filename='message_topics_50.csv'
# message_stats_filename='message_stats.csv'
message_stats_filename='message_stats_50.csv'
stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves',
              'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
              'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
              'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
              'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about',
              'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up',
              'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
              'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
              'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'xa']

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
            # filename=prefix+ n+'_'+c + suffix
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
    print data[:1]
    print data.shape

    return data

def get_tokens(desc):
    return [ word for word in tokenizer.tokenize( desc ) if word not in stopwords ]

def read_message_data():
    topics = pd.read_csv(root_dir+results_sub_dir+message_filename)
    stats = pd.read_csv(root_dir+results_sub_dir+message_stats_filename)
    topics_cp_filename = 'topics_cp.csv'
    # topics_cp_filename = 'topics_t50ll.csv'
    topics_cp = pd.read_csv(root_dir+results_sub_dir+topics_cp_filename)
    # topics_weights = topics_cp.groupby(['category'])[['weight']].sum()
    # topics_weights.columns = ['category_weight']
    # topics_cp = pd.merge(topics_cp, topics_weights, left_on='category', right_index=True, how='inner')
    # print topics_cp[:3]
    # topics_cp['weight'] = topics_cp['weight']/topics_cp['category_weight']

    stats['words'] = stats['message'].apply( partial( get_tokens))
    # topics = topics.groupby('id' )['group_norm'].transform(lambda x: (x  / x.sum()))
    print topics[:5]

    # print topics.shape
    # print stats.columns
    # print stats.shape
    print topics[:1]
    print stats[:1]
    print topics_cp[:1]

    # print topics[topics['group_id']== 1046055923301711873]

    return topics, stats, topics_cp

def calc_contagious_score(data):
    def func_calc_score(row, cols):
        suffix = '_inf0'
        [ angle_fr, angle_lf, angle_lr]  = [ np.arccos(row[csim+suffix])/np.pi for csim in ['dot_fr', 'dot_lf', 'dot_lr'] ]
        score =  angle_lr / angle_lf
        # score = row['csim_lf'+suffix]
        score = row['csim_lf'+suffix] - row['csim_lr'+suffix]
        return score
    columns = list(data.columns.values)
    data['score'] = data.apply(partial(func_calc_score, cols=columns) , axis=1)
    data['score_rank'] = data['score'].rank(ascending=False)

    print 'data.shape: '
    print data.shape
    # data.drop( data[ data['score'] < 0 ].index , inplace=True)
    data.sort_values(by=['score'], inplace=True, ascending=False)
    # data.drop( data[1000:].index , inplace=True)
    print data.shape
    data['score'] = data.score.transform(lambda x: (x - x.min()) /  (x.max()-x.min()) )
    # data[-1000:]['score'] = 0

    min_ = data.score.min()
    print 'min score: ', min_
    # data['score'] = data['score'] - min_
    data.sort_values(by=['score'], inplace=True, ascending=False)
    # data['score'][-1000:] = 0
    print data.score[:10]
    print data.score[-10:]
    print '--------------------------'
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
    print '--------------------------'

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




def calc_message_score2(topics_cp, data, stats):
    print 'calc_message_score2...'
    def message_score(message, words_score):
        # try:
        vals = []
        val = 0
        # count = 0
        # print 'length: ', len(message) , ' , ' , message
        for word in message:
            # val += words_score.loc[(words_score['term'] == word)]['word_score']
            # print words_score.loc[word]
            # print word in words_score.index
            if word in words_score.index:
                vals.append(words_score.loc[word].values[0])
                val += words_score.loc[word].values[0]
                # print word , ' , ', words_score.loc[word].values[0]
                # count+=1
        # except:
        #     print '-----error-----'
        #     print type(message)
        vals = sorted(vals)
        # val = np.sum(vals[-5:])
        return val #*1.0/count if count != 0 else 0
    topics_cp = pd.merge(topics_cp, data, left_on=['category'], right_on=['topic'], how='inner')

    print '----> max: ', topics_cp.score.max() , ' , min: ', topics_cp.score.min()
    print topics_cp.shape
    #topics_cp.drop( topics_cp[ topics_cp['score'] < 0 ].index , inplace=True)
    print topics_cp.shape
    print topics_cp.columns
    # print 'unique user_id: ', len(list(topics_cp.user_id.unique()))
    topics_cp['word_score'] =topics_cp['weight'] * topics_cp['score']
    top10_scores = topics_cp.sort_values(['term','word_score'],ascending=False).groupby('term').head(10)
    # print top10_scores.shape
    print top10_scores[:20][['term','word_score']]
    words_score = topics_cp.groupby(['term'])[['word_score']].sum()
    words_score.sort_values(by=['word_score'], inplace=True, ascending=False)
    print words_score[['word_score']][:50]
    # words_score.reset_index(inplace=True)
    print '====='
    # print words_score[['word_score']][-20:]
    # print len(list(topics_cp.term.unique()))
    print words_score.shape
    stats['message_score'] = stats['words'].apply(partial(message_score, words_score=words_score))
    stats['word_count'] = stats['words'].apply(lambda x: len(x))
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



    msg_df  = pd.merge(stats, user_df, left_on=['user_id'], right_on=['user_id'], how='inner')
    print msg_df.columns
    print msg_df.shape
    print msg_df[[ 'message_score', 'user_score']][:10]

    #msg_df = msg_df[((msg_df['retweet_count']>0) & (msg_df['favorite_count']>0))]
    #print msg_df.shape
    #print sum_[:5]
    msg_df['message_score_dif'] = msg_df['message_score'] - msg_df['user_score']
    msg_df['retweet_dif'] = msg_df['retweet_count'] - msg_df['user_retweet']
    msg_df['favorite_dif'] = msg_df['favorite_count'] - msg_df['user_favorite']
    print "-------"
    print msg_df[[ 'message_score_dif', 'message_score', 'user_score']][:10]
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


    norm_msg_df = msg_df.groupby('user_id')['message_score', 'retweet_count', 'favorite_count', 'word_count'].transform(lambda x: (x - x.min()) / (x.max()-x.min()))
    norm_msg_df['user_id'] = msg_df['user_id']
    print norm_msg_df[:10]
    print msg_df[:10]
    print "-------"
    print norm_msg_df.corr(method='spearman')[[ 'message_score', 'retweet_count', 'favorite_count', 'word_count']]
    print "=========="

    users = list(user_df.user_id.unique())
    scores_pearson = []
    scores_spearman = []
    scores = []

    for u in users:
        # person = msg_df.loc[msg_df['user_id'] == u]
        # corr_pearson = person.corr(method='pearson')['message_score_dif'][['favorite_dif', 'retweet_dif']].values
        # corr_spearman  = person.corr(method='spearman')['message_score_dif'][['favorite_dif', 'retweet_dif']].values
        # scores_pearson.append(corr_pearson)
        # scores_spearman.append(corr_spearman)
        # print u, ' --- ', person.corr(method='spearman')['favorite_dif'][['message_score_dif']].values[0] , \
        #     ' ---- ', person.corr(method='spearman')['retweet_dif'][['message_score_dif']].values[0]


        person = norm_msg_df.loc[norm_msg_df['user_id'] == u]
        X = person[[ 'message_score', 'word_count' ]]
        Y = person[['retweet_count', 'favorite_count']]
        score_ret = learn(X.values, Y.values[:,0])
        score_ret_base = learn(X.values[:,1].reshape(-1, 1), Y.values[:,0])
        score_fav = learn(X.values, Y.values[:,1])
        score_fav_base = learn(X.values[:,1].reshape(-1, 1), Y.values[:,1])
        scores.append([score_ret, score_ret_base, score_fav, score_fav_base])
        print u, ' --- ' , [score_ret, score_ret_base, score_fav, score_fav_base]

    print scores_pearson
    print 'mean_pearson: ' , np.mean(scores_pearson, 0)
    print 'mean_spearman: ' , np.mean(scores_spearman, 0)
    print 'mean_score: ' , np.mean(scores, 0)

    # normalized_msg_df = msg_df.groupby('user_id')['message_score', 'retweet_count', 'favorite_count', 'word_count'].transform(lambda x: (x - x.mean()) / x.std())




    X = norm_msg_df[[ 'message_score', 'word_count' ]]
    Y = norm_msg_df[['retweet_count', 'favorite_count']]
    print X.shape
    print Y.shape
    return X, Y



def calc_message_score3(topics, data, stats):
    print 'calc_message_score2...'

    data.reset_index(inplace=True)
    print 'data.shape: ', data.shape
    print 'data.columns: ', data.columns
    print 'unique topics in data: ', len(list(data.topic.unique()))
    print 'topics:'
    print topics.shape
    print topics[topics['group_id']== 1046055923301711873].shape
    topics = pd.merge(topics, data, left_on=['feat'], right_on=['topic'], how='inner')
    print '----> max: ', topics.score.max() , ' , min: ', topics.score.min()
    print topics.shape
    print topics[topics['group_id']== 1046055923301711873].shape
    #topics_cp.drop( topics_cp[ topics_cp['score'] < 0 ].index , inplace=True)
    topics.sort_values(['group_id'], ascending=False, inplace=True)
    # print topics[['id', 'group_id', 'feat', 'value', 'group_norm', 'score']][:200]
    print topics.columns
    # print 'unique user_id: ', len(list(topics_cp.user_id.unique()))
    topics['score_gn'] =topics['group_normalized'] * topics['score']
    print 'sum(score_gn):::: ' , np.sum(topics[topics['group_id']== 1046055923301711873].score_gn.values)

    top10_scores = topics.sort_values(['group_id','score_gn'],ascending=False).groupby('group_id').head(10)
    msg_score_df = top10_scores.groupby(['group_id'])[['score_gn']].sum()
    # print 'msg_score_df[1046055923301711873]: ' , msg_score_df[1046055923301711873]
    msg_score_df.columns = ['message_score']
    msg_score_df.sort_values(by=['message_score'], inplace=True, ascending=False)
    # print msg_score[['message_score']][:50]
    print '====='
    # stats['message_score'] = stats['words'].apply(partial(message_score, words_score=words_score))
    stats = pd.merge(stats, msg_score_df, left_on='message_id', right_on='group_id', how='inner')
    stats['word_count'] = stats['words'].apply(lambda x: len(x))
    # print 'stats:'
    # print stats[:10]

    print stats.columns
    print topics.columns
    print data.columns

    # user_df= stats.groupby(['user_id'])[['retweet_count']].mean()
    # user_df['favorite'] = stats.groupby(['user_id'])[['favorite_count']].mean()
    # user_df['count'] = stats.groupby(['user_id'])[['favorite_count']].count()
    # user_df.columns = ['user_retweet', 'user_favorite', 'user_count']
    # print ('user_df:')
    # print user_df[:5]
    #
    #
    # print stats[:10]
    # # temp = msg_df.groupby(['group_id', 'user_id'])[['score_gn']].sum()
    # user_df['user_score'] = stats.groupby(['user_id'])[['message_score']].mean()
    # user_df.reset_index(inplace=True)




    # msg_df  = pd.merge(stats, user_df, left_on=['user_id'], right_on=['user_id'], how='inner')
    # print msg_df.columns
    # print msg_df.shape
    # print msg_df[[ 'message_score', 'user_score']][:10]

    #msg_df = msg_df[((msg_df['retweet_count']>0) & (msg_df['favorite_count']>0))]
    #print msg_df.shape
    #print sum_[:5]
    # msg_df['message_score_dif'] = msg_df['message_score'] - msg_df['user_score']
    # msg_df['retweet_dif'] = msg_df['retweet_count'] - msg_df['user_retweet']
    # msg_df['favorite_dif'] = msg_df['favorite_count'] - msg_df['user_favorite']
    # print "-------"
    # print msg_df[[ 'message_score_dif', 'message_score', 'user_score']][:10]
    # print "-------"
    # print msg_df.corr(method='spearman')[[ 'message_score_dif', 'message_score', 'user_score', 'word_count']]
    # print "=========="

    stats['fav_ret'] = stats['retweet_count'] + stats['favorite_count']
    norm_msg_df = stats.groupby('user_id')['message_score', 'retweet_count', 'favorite_count','fav_ret', 'word_count'].transform(lambda x: (x - x.min()) / (x.max()-x.min()))
    norm_msg_df['user_id'] = stats['user_id']
    print norm_msg_df[:10]
    print stats[:10]
    print "-------"
    print 'norm:'
    print norm_msg_df.corr(method='spearman')[[ 'message_score', 'retweet_count', 'favorite_count', 'fav_ret', 'word_count']]
    print 'stats:'
    print stats.corr(method='spearman')[[ 'message_score', 'retweet_count', 'favorite_count', 'fav_ret', 'word_count']]
    print "=========="

    users = list(stats.user_id.unique())
    print 'len(users): ', len(users)
    scores_pearson = []
    scores_spearman = []
    scores = []

    for u in users:
        person = stats.loc[stats['user_id'] == u]
        corr_pearson = person.corr(method='pearson')['message_score'][['retweet_count', 'favorite_count']].values
        corr_spearman  = person.corr(method='spearman')['message_score'][['retweet_count', 'favorite_count']].values
        scores_pearson.append(corr_pearson)
        scores_spearman.append(corr_spearman)
        print u, ' --- ', person.corr(method='spearman')['retweet_count'][['message_score']].values[0] , \
            ' ---- ', person.corr(method='spearman')['favorite_count'][['message_score']].values[0]


        # person = norm_msg_df.loc[norm_msg_df['user_id'] == u]
        # X = person[[ 'message_score', 'word_count' ]]
        # Y = person[['retweet_count', 'favorite_count']]
        # score_ret = learn(X.values, Y.values[:,0])
        # score_ret_base = learn(X.values[:,1].reshape(-1, 1), Y.values[:,0])
        # score_fav = learn(X.values, Y.values[:,1])
        # score_fav_base = learn(X.values[:,1].reshape(-1, 1), Y.values[:,1])
        # scores.append([score_ret, score_ret_base, score_fav, score_fav_base])
        # print u, ' --- ' , [score_ret, score_ret_base, score_fav, score_fav_base]

    print scores_pearson
    print 'mean_pearson: ' , np.mean(scores_pearson, 0)
    print 'mean_spearman: ' , np.mean(scores_spearman, 0)
    print 'mean_score: ' , np.mean(scores, 0)



    X = norm_msg_df[[ 'message_score', 'word_count' ]]
    Y = norm_msg_df[['retweet_count', 'favorite_count']]
    print X.shape
    print Y.shape
    return X, Y


def learn(x,y):
    print 'learn...'
    print x.shape, ' , ', y.shape
    # x = X[:,:]
    # y = Y[:,0]
    # print x[:5]
    # print y[:5]
    reg = LinearRegression().fit(x, y) #max_iter=100, tol=1e-4)
    reg.fit(x, y)
    ypred = reg.predict(x)
    rmse1 = np.sqrt(mean_squared_error(y, ypred))
    print 'rmse: ', rmse1
    # print 'coef: ', reg.coef_, ' , ', reg.intercept_

    val = np.mean(y)
    # print 'val: ', val
    ypred2 = [ val for i in y ]
    rmse2 = np.sqrt(mean_squared_error(y, ypred2))
    print 'rmse: ', rmse2

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

X, Y = calc_message_score3(topics, data, stats)

learn(X.values, Y.values[:,0])
learn(X.values[:,1].reshape(-1, 1), Y.values[:,0])
learn(X.values, Y.values[:,0])
learn(X.values[:,1].reshape(-1, 1), Y.values[:,0])



