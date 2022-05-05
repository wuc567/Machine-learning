# -*- coding:utf-8 -*-
"""
Description: A python 2.7 implementation of gcForestCS proposed in [1]. A demo implementation of gcForest library as well as some demo client scripts to demostrate how to use the code. The implementation is flexible enough for modifying the model or
fit your own datasets.
Reference: [1] M. Pang, K. M. Ting, P. Zhao, and Z.-H. Zhou. Improving deep forest by confidence screening. In ICDM-2018.  (http://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm18.pdf)
Requirements: This package is developed with Python 2.7, please make sure all the demendencies are installed, which is specified in requirements.txt
ATTN: This package is free for academic usage. You can run it at your own risk. For other purposes, please contact Prof. Zhi-Hua Zhou(zhouzh@lamda.nju.edu.cn)
ATTN2: This package is developed by Mr. Ming Pang(pangm@lamda.nju.edu.cn), which is based on the gcForest package (http://lamda.nju.edu.cn/code_gcForest.ashx). The readme file and demo roughly explains how to use the codes. For any problem concerning the codes, please feel free to contact Mr. Pang.
"""
import numpy as np
import os
import os.path as osp
import pickle
from matplotlib.font_manager import *
from ..estimators import get_estimator_kfold
from ..utils.config_utils import get_config_value
from ..utils.log_utils import get_logger
from ..utils.metrics import accuracy_pb
import matplotlib.pyplot as plt
import psutil
from scipy.optimize import leastsq

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

LOGGER = get_logger('gcforest.cascade.cascade_classifier_cs')


def check_dir(path):
    d = osp.abspath(osp.join(path, osp.pardir))
    if not osp.exists(d):
        os.makedirs(d)


def calc_accuracy(y_true, y_pred, name, prefix=""):
    acc = 100. * np.sum(np.asarray(y_true) == y_pred) / max(len(y_true), 1.0)
    LOGGER.info('{}Accuracy({})={:.2f}%'.format(prefix, name, acc))
    return acc


def get_opt_layer_id(acc_list):
    """ Return layer id with max accuracy on training data """
    opt_layer_id = np.argsort(-np.asarray(acc_list), kind='mergesort')[0]
    return opt_layer_id


def get_more2_y(y_all, y_bool, num_atleast_each_class=3):
    y = y_all[y_bool]
    y_set = set(y_all)
    del_y = np.zeros(len(y_all))
    del_y[~y_bool] = 1
    for yi in y_set:
        if np.sum(y == yi) < num_atleast_each_class:
            tmp_index = np.arange(len(y_all))[y_all == yi]
            tmp_index = tmp_index[np.random.permutation(len(tmp_index))[:num_atleast_each_class]]
            del_y[tmp_index] = 0
    return del_y < 1

def getAutoThresh1(y_proba, y, part_decay, bin_size):
    import numpy as np
    import pandas as pd
    import matplotlib

    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator
    from matplotlib import rcParams
    matplotlib.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Times New Roman'],
    })  # 设置全局字体
    # 定义自定义字体，文件名从1.b查看系统中文字体中来
    myfont = FontProperties(fname='C:\Windows\Fonts\STSONG.ttf')
    # 解决负号'-'显示为方块的问题
    matplotlib.rcParams['axes.unicode_minus'] = False

    right_bool = (np.asarray(y) == np.argmax(y_proba, axis=1))
    n_inst = y_proba.shape[0]
    acc_now = np.sum(right_bool)/float(n_inst)
    acc_require = 1-(1-acc_now)/float(part_decay)
    LOGGER.info("Training accuracy now is {} and accuracy part1 needs to get {} ({} times)".format(acc_now, acc_require, part_decay))
    num = int(n_inst/bin_size)
    print(bin_size)
    CX_max = np.max(y_proba, axis=1)
    index_sort = np.argsort(-CX_max)

    acc_all_block = np.zeros(num)
    num_in_block = int(n_inst/num)

    for i in range(0, num):
        for j in range(i*num_in_block, (i+1)*num_in_block):
            if right_bool[index_sort[j]]:
                acc_all_block[i] += 1
        acc_all_block[i] = acc_all_block[i]/num_in_block
    for i in range(0, num):
        if acc_all_block[i] < acc_require:
            flag = i
            break
    if flag == 0:
        return 1

    CX_max = np.max(y_proba, axis=1)
    index_sort = np.argsort(-CX_max)
    acc_all = np.zeros(n_inst)
    if right_bool[index_sort[0]]:
        acc_all[0] = 1
    for i in range(1, n_inst):
        if right_bool[index_sort[i]]:
            acc_all[i] = acc_all[i - 1] + 1
        else:
            acc_all[i] = acc_all[i - 1]
    num_all = np.arange(1, n_inst + 1)
    acc_all /= num_all

    index_sat = acc_all > acc_require
    if np.sum(index_sat) < 1:
        num_part2 = 0
    else:
        num_part2 = np.max(np.arange(n_inst)[index_sat])
        print(num_part2/100)

    # #拟合曲线
    # x = np.arange(0, num)
    # y = acc_all_block
    #leastsq
    """def func(p,x):
        a,b,c=p
        return a*x**2+b*x+c
    def error(p,x,y):
        return func(p,x)-y
    p0 = [1,1,1]

    para = leastsq(error,p0,args=(x,y))
    print(para)
    a,b,c = para[0]
    y_s = a*x**2+b*x+c"""
    #np.polyfit
    # f1 = np.polyfit(x,y,10)
    # p1 = np.poly1d(f1)
    # y_s = p1(x)


    # #画图
    # x = np.arange(0,int(acc_all_block.shape[0]),1)
    # ax1 = plt.subplot(1, 1, 1, facecolor='white')
    # plt.xticks(fontsize=13)
    # plt.yticks(fontsize=13)
    # ax1.set_xlabel(u'箱子编号',fontsize=15, fontproperties=myfont)
    # ax1.set_ylabel(u'训练准确率/%',fontsize=15, fontproperties=myfont)
    # ax1.set_xlim(1,acc_all_block.shape[0])
    # ax1.set_ylim(0,110)
    #
    # ax1.plot(x, acc_all_block*100)
    # # plt.plot(x,y_s,color = "red")
    # line1= ax1.vlines(x=int(num_part2/100), ymin=50, ymax=110, linestyles='--', colors='g',label="gcForestcs")
    # line2= ax1.vlines(x=flag, ymin=70, ymax=110, linestyles='--', colors='r', label="DBC-Forest")
    # print("DBC-Forest is {}".format(flag))
    # print("gcForestCS is {}".format(int(num_part2/100)))
    # plt.legend(handles=[line1, line2], labels=['gcForestcs', 'DBC-Forest'], fontsize=13, loc="lower left")
    # plt.savefig('E:\Backup\桌面\大论文\图片\分箱置信度筛选森林\阈值\compare of layer.jpg', dpi=400, bbox_inches='tight')
    # plt.show()


    return CX_max[index_sort[flag*num_in_block-1]]

def decide_thresh1(y_train_proba_li, y_train, part_decay,layer_id, bin_size):
    # LOGGER.info("CX infomation using y_proba_li instead of X_cur")
    CX_train = np.max(y_train_proba_li, axis=1)

    CX_thresh = getAutoThresh1(y_train_proba_li, y_train, part_decay, bin_size=bin_size)
    LOGGER.info("#instances = {}, , CX threshold: {}".format(y_train_proba_li.shape[0], CX_thresh))
    return CX_thresh, CX_train

def getAutoThresh(y_proba, y, part_decay):
    right_bool = (np.asarray(y) == np.argmax(y_proba, axis=1))
    n_inst = y_proba.shape[0]
    acc_now = np.sum(right_bool)/float(n_inst)
    acc_require = 1-(1-acc_now)/float(part_decay)
    LOGGER.info("Training accuracy now is {} and accuracy part1 needs to get {} ({} times)".format(acc_now,acc_require,part_decay))

    CX_max = np.max(y_proba, axis=1)
    index_sort = np.argsort(-CX_max)
    acc_all = np.zeros(n_inst)
    if right_bool[index_sort[0]]:
        acc_all[0] = 1
    for i in range(1,n_inst):
        if right_bool[index_sort[i]]:
            acc_all[i]=acc_all[i-1]+1
        else:
            acc_all[i]=acc_all[i-1]
    num_all = np.arange(1, n_inst+1)
    acc_all /= num_all

    index_sat = acc_all>acc_require
    if np.sum(index_sat) < 1:
        num_part2 = 0
    else:
        num_part2 = np.max(np.arange(n_inst)[index_sat])
    return n_inst-num_part2-1


def decide_thresh(y_train_proba_li, y_train, part_decay):
    # LOGGER.info("CX infomation using y_proba_li instead of X_cur")
    CX_train = np.max(y_train_proba_li, axis=1)

    if 0 < part_decay <= 1:
        num_train_part1 = int(y_train_proba_li.shape[0]*part_decay)-1
    elif part_decay > 1:
        num_train_part1 = getAutoThresh(y_train_proba_li, y_train, part_decay)
    else:
        LOGGER.info("Wrong input of part decay which should be (0,1] or >1")
        assert False

    CX_thresh = np.sort(CX_train)[num_train_part1]
    LOGGER.info("#instances = {}, num_thresh {}, CX threshold: {}".format(y_train_proba_li.shape[0], num_train_part1, CX_thresh))
    return CX_thresh, CX_train

class CascadeClassifier_th(object):
    def __init__(self, ca_config):
        """
        Parameters (ca_config)
        ----------
        early_stopping_rounds: int
            when not None , means when the accuracy does not increase in early_stopping_rounds, the cascade level will stop automatically growing
        max_layers: int
            maximum number of cascade layers allowed for exepriments, 0 means use Early Stoping to automatically find the layer number
        n_classes: int
            Number of classes
        est_configs:
            List of CVEstimator's config
        look_indexs_cycle (list 2d): default=None
            specification for layer i, look for the array in look_indexs_cycle[i % len(look_indexs_cycle)]
            defalut = None <=> [range(n_groups)]
            .e.g.
                look_indexs_cycle = [[0,1],[2,3],[0,1,2,3]]
                means layer 1 look for the grained 0,1; layer 2 look for grained 2,3; layer 3 look for every grained, and layer 4 cycles back as layer 1
        data_save_rounds: int [default=0]
        data_save_dir: str [default=None]
            each data_save_rounds save the intermidiate results in data_save_dir
            if data_save_rounds = 0, then no savings for intermidiate results
        """
        self.ca_config = ca_config
        self.early_stopping_rounds = self.get_value("early_stopping_rounds", None, int, required=True)
        self.max_layers = self.get_value("max_layers", 0, int)
        self.n_classes = self.get_value("n_classes", None, int, required=True)
        self.est_configs = self.get_value("estimators", None, list, required=True)
        self.look_indexs_cycle = self.get_value("look_indexs_cycle", None, list)
        self.random_state = self.get_value("random_state", None, int)
        # self.data_save_dir = self.get_value("data_save_dir", None, basestring)
        self.data_save_dir = ca_config.get("data_save_dir", None)
        self.data_save_rounds = self.get_value("data_save_rounds", 0, int)
        if self.data_save_rounds > 0:
            assert self.data_save_dir is not None, "data_save_dir should not be null when data_save_rounds>0"
        self.eval_metrics = [("predict", accuracy_pb)]
        self.estimator2d = {}
        self.opt_layer_num = -1
        self.bin_size = self.get_value("bin_size", 100, int)

        self.part_decay = self.get_value("part_decay", 3, int)
        self.estimators_enlarge = self.get_value("estimators_enlarge", True, bool)
        self.conf_thresh = []
        self.train_num_level = []
        self.train_decay_level = []
        self.train_index1_level = []
        self.train_index2_level = []
        self.train_acc1_level = []
        self.train_acc2_level = []

        self.test_num_level = []
        self.test_decay_level = []
        self.test_index1_level = []
        self.test_index2_level = []
        self.test_acc1_level = []
        self.test_acc2_level = []

        # LOGGER.info("\n" + json.dumps(ca_config, sort_keys=True, indent=4, separators=(',', ':')))

    @property
    def n_estimators_1(self):
        # estimators of one layer
        return len(self.est_configs)

    def get_value(self, key, default_value, value_types, required=False):
        return get_config_value(self.ca_config, key, default_value, value_types,
                required=required, config_name="cascade")

    def _set_estimator(self, li, ei, est):
        if li not in self.estimator2d:
            self.estimator2d[li] = {}
        self.estimator2d[li][ei] = est

    def _get_estimator(self, li, ei):
        return self.estimator2d.get(li, {}).get(ei, None)

    def _init_estimators_enlarge(self, li, ei, ratio_decay):
        est_args = self.est_configs[ei].copy()
        est_name = "layer_{} - estimator_{} - {}_folds".format(li, ei, est_args["n_folds"])
        # n_folds
        n_folds = int(est_args["n_folds"])
        est_args.pop("n_folds")
        # est_type
        est_type = est_args["type"]
        est_args.pop("type")
        est_args["n_estimators"] = int(est_args["n_estimators"]/ratio_decay)
        # random_state
        if self.random_state is not None:
            random_state = (self.random_state + hash("[estimator] {}".format(est_name))) % 1000000007
        else:
            random_state = None
        return get_estimator_kfold(est_name, n_folds, est_type, est_args, random_state=random_state)

    def _init_estimators(self, li, ei):
        est_args = self.est_configs[ei].copy()
        est_name = "layer_{} - estimator_{} - {}_folds".format(li, ei, est_args["n_folds"])
        # n_folds
        n_folds = int(est_args["n_folds"])
        est_args.pop("n_folds")
        # est_type
        est_type = est_args["type"]
        est_args.pop("type")
        # random_state
        if self.random_state is not None:
            random_state = (self.random_state + hash("[estimator] {}".format(est_name))) % 1000000007
        else:
            random_state = None
        return get_estimator_kfold(est_name, n_folds, est_type, est_args, random_state=random_state)

    def _check_look_indexs_cycle(self, X_groups, is_fit):
        # check look_indexs_cycle
        n_groups = len(X_groups)
        if is_fit and self.look_indexs_cycle is None:
            look_indexs_cycle = [list(range(n_groups))]
        else:
            look_indexs_cycle = self.look_indexs_cycle
            for look_indexs in look_indexs_cycle:
                if np.max(look_indexs) >= n_groups or np.min(look_indexs) < 0 or len(look_indexs) == 0:
                    raise ValueError("look_indexs doesn't match n_groups!!! look_indexs={}, n_groups={}".format(
                        look_indexs, n_groups))
        if is_fit:
            self.look_indexs_cycle = look_indexs_cycle
        return look_indexs_cycle

    def _check_group_dims(self, X_groups, is_fit):
        if is_fit:
            group_starts, group_ends, group_dims = [], [], []
        else:
            group_starts, group_ends, group_dims = self.group_starts, self.group_ends, self.group_dims
        n_datas = X_groups[0].shape[0]
        X = np.zeros((n_datas, 0), dtype=X_groups[0].dtype)
        for i, X_group in enumerate(X_groups):
            assert(X_group.shape[0] == n_datas)
            X_group = X_group.reshape(n_datas, -1)
            if is_fit:
                group_dims.append( X_group.shape[1] )
                group_starts.append(0 if i == 0 else group_ends[i - 1])
                group_ends.append(group_starts[i] + group_dims[i])
            else:
                assert(X_group.shape[1] == group_dims[i])
            X = np.hstack((X, X_group))
        if is_fit:
            self.group_starts, self.group_ends, self.group_dims = group_starts, group_ends, group_dims
        return group_starts, group_ends, group_dims, X
    def confidence_screening(self, y_train_proba_li, y_train, train_index_now, layer_id):
        CX_thresh, CX_train = decide_thresh1(y_train_proba_li=y_train_proba_li, y_train=y_train,part_decay=self.part_decay,layer_id=layer_id, bin_size = self.bin_size)
        #print("part_decay is {}".format(self.part_decay))
        #CX_thresh, CX_train = decide_thresh1(y_train_proba_li=y_train_proba_li, y_train=y_train, part_decay=2)
        self.conf_thresh.append(CX_thresh)###门阀值
        train_bool_1part = CX_train >= CX_thresh###选取大于门阀值的样本
        train_bool_2part = ~train_bool_1part###剩下的样本
        num_atleast_each_class = self.est_configs[0]["n_folds"]
        train_bool_2part = get_more2_y(y_train, train_bool_2part, num_atleast_each_class)
        train_bool_1part = ~train_bool_2part

        train_index1_now = train_index_now[train_bool_1part]
        train_index2_now = train_index_now[train_bool_2part]
        self.train_index1_level.append(train_index1_now)
        self.train_index2_level.append(train_index2_now)
        self.train_num_level.append(np.sum(train_bool_2part))
        part_decay_i = self.train_num_level[layer_id] / float(self.train_num_level[layer_id - 1])
        self.train_decay_level.append(part_decay_i)
        LOGGER.info("In layer {}, train num is {}, part decay is {}".format(layer_id,
                                                                            self.train_num_level[layer_id],
                                                                            part_decay_i))
        train_1part_acc = calc_accuracy(y_train[train_bool_1part],
                                        np.argmax(y_train_proba_li[train_bool_1part, :], axis=1),
                                        'layer_{} - train.1part({})'.format(layer_id - 1,
                                                                            np.sum(train_bool_1part)))
        train_2part_acc = calc_accuracy(y_train[train_bool_2part],
                                        np.argmax(y_train_proba_li[train_bool_2part, :], axis=1),
                                        'layer_{} - train.2part({})'.format(layer_id - 1,
                                                                            np.sum(train_bool_2part)))
        self.train_acc1_level.append(train_1part_acc)
        self.train_acc2_level.append(train_2part_acc)
        num_train_correct_add = train_1part_acc / 100.0 * np.sum(train_bool_1part)
        return train_bool_2part, num_train_correct_add

    def confidence_screening_test(self, y_test_proba_li, y_test, test_index_now, layer_id):###返回置信度筛选后的数据
        CX_test = np.max(y_test_proba_li, axis=1)
        CX_thresh = self.conf_thresh[-1]
        test_bool_1part = CX_test >= CX_thresh
        test_bool_2part = ~test_bool_1part

        test_index1_now = test_index_now[test_bool_1part]
        test_index2_now = test_index_now[test_bool_2part]
        self.test_index1_level.append(test_index1_now)
        self.test_index2_level.append(test_index2_now)
        self.test_num_level.append(np.sum(test_bool_2part))
        part_decay_i = self.test_num_level[layer_id] / float(self.test_num_level[layer_id - 1])
        self.test_decay_level.append(part_decay_i)
        LOGGER.info("In layer {}, test num is {}, part decay is {}".format(layer_id,
                                                                            self.test_num_level[layer_id],
                                                                            part_decay_i))

        test_1part_acc = calc_accuracy(y_test[test_bool_1part],
                                        np.argmax(y_test_proba_li[test_bool_1part, :], axis=1),
                                        'layer_{} - test.1part({})'.format(layer_id - 1,
                                                                            np.sum(test_bool_1part)))###置信度筛选的正确率
        test_2part_acc = calc_accuracy(y_test[test_bool_2part],
                                        np.argmax(y_test_proba_li[test_bool_2part, :], axis=1),
                                        'layer_{} - test.2part({})'.format(layer_id - 1,
                                                                            np.sum(test_bool_2part)))###剩下的正确率
        self.test_acc1_level.append(test_1part_acc)
        self.test_acc2_level.append(test_2part_acc)
        num_test_correct_add = test_1part_acc / 100.0 * np.sum(test_bool_1part)
        return test_bool_2part, num_test_correct_add

    def fit_transform(self, X_groups_train, y_train, X_groups_test, y_test, stop_by_test=False, train_config=None):
        """
        fit until the accuracy converges in early_stop_rounds
        stop_by_test: (bool)
            When X_test, y_test is validation data that used for determine the opt_layer_id,
            use this option
        """

        if train_config is None:
            from ..config import GCTrainConfig
            train_config = GCTrainConfig({})
        data_save_dir = train_config.data_cache.cache_dir or self.data_save_dir

        is_eval_test = "test" in train_config.phases
        if not type(X_groups_train) == list:
            X_groups_train = [X_groups_train]
        if is_eval_test and not type(X_groups_test) == list:
            X_groups_test = [X_groups_test]
        LOGGER.info("X_groups_train.shape={},y_train.shape={},X_groups_test.shape={},y_test.shape={}".format(
            [xr.shape for xr in X_groups_train], y_train.shape,
            [xt.shape for xt in X_groups_test] if is_eval_test else "no_test", y_test.shape if is_eval_test else "no_test"))

        # check look_indexs_cycle
        look_indexs_cycle = self._check_look_indexs_cycle(X_groups_train, True)
        if is_eval_test:
            self._check_look_indexs_cycle(X_groups_test, False)

        # check groups dimension
        group_starts, group_ends, group_dims, X_train = self._check_group_dims(X_groups_train, True)
        if is_eval_test:
            _, _, _, X_test = self._check_group_dims(X_groups_test, False)
        else:
            X_test = np.zeros((0, X_train.shape[1]))
        LOGGER.info("group_dims={}".format(group_dims))
        LOGGER.info("group_starts={}".format(group_starts))
        LOGGER.info("group_ends={}".format(group_ends))
        LOGGER.info("X_train.shape={},X_test.shape={}".format(X_train.shape, X_test.shape))

        n_trains = X_groups_train[0].shape[0]
        n_tests = X_groups_test[0].shape[0] if is_eval_test else 0

        n_classes = self.n_classes
        assert n_classes == len(np.unique(y_train)), "n_classes({}) != len(unique(y)) {}".format(n_classes, np.unique(y_train))
        train_acc_list = []
        test_acc_list = []
        # X_train, y_train, X_test, y_test
        opt_datas = [None, None, None, None]
        try:
            # probability of each cascades's estimators
            X_proba_train = np.zeros((n_trains, n_classes * self.n_estimators_1), dtype=np.float32)
            X_proba_test = np.zeros((n_tests, n_classes * self.n_estimators_1), dtype=np.float32)
            X_cur_train, X_cur_test = None, None
            layer_id = 0

            self.train_num_level.append(n_trains)
            self.train_decay_level.append(1.0)
            train_index_now = np.arange(n_trains)
            self.train_index1_level.append(np.array([]))
            self.train_index2_level.append(train_index_now)
            num_train_correct = 0

            if n_tests > 0:
                self.test_num_level.append(n_tests)
                self.test_decay_level.append(1.0)
                test_index_now = np.arange(n_tests)
                self.test_index1_level.append(np.array([]))
                self.test_index2_level.append(test_index_now)
                num_test_correct = 0

            while 1:
                if self.max_layers > 0 and layer_id >= self.max_layers:
                    break
                # Copy previous cascades's probability into current X_cur
                if layer_id == 0:
                    # first layer not have probability distribution
                    X_cur_train = np.zeros((n_trains, 0), dtype=np.float32)
                    X_cur_test = np.zeros((n_tests, 0), dtype=np.float32)
                else:
                    X_cur_train = X_proba_train.copy()
                    X_cur_test = X_proba_test.copy()

                # Confidence screening
                if layer_id > 0:
                    train_bool_2part, num_train_correct_add = self.confidence_screening(
                        y_train_proba_li, y_train, train_index_now, layer_id)
                    train_index_now = train_index_now[train_bool_2part]
                    X_cur_train = X_cur_train[train_bool_2part, :]
                    X_proba_train = X_proba_train[train_bool_2part, :]
                    X_train = X_train[train_bool_2part, :]
                    y_train = y_train[train_bool_2part]
                    num_train_correct += num_train_correct_add
                    if n_tests > 0:
                        test_bool_2part, num_test_correct_add = self.confidence_screening_test(
                            y_test_proba_li, y_test, test_index_now, layer_id)
                        test_index_now = test_index_now[test_bool_2part]
                        X_cur_test = X_cur_test[test_bool_2part, :]
                        X_proba_test = X_proba_test[test_bool_2part, :]
                        X_test = X_test[test_bool_2part, :]
                        y_test = y_test[test_bool_2part]
                        num_test_correct += num_test_correct_add
                # Stack data that current layer needs in to X_cur
                look_indexs = look_indexs_cycle[layer_id % len(look_indexs_cycle)]
                for _i, i in enumerate(look_indexs):
                    X_cur_train = np.hstack((X_cur_train, X_train[:, group_starts[i]:group_ends[i]]))
                    X_cur_test = np.hstack((X_cur_test, X_test[:, group_starts[i]:group_ends[i]]))
                LOGGER.info("[layer={}] look_indexs={}, X_cur_train.shape={}, X_cur_test.shape={}".format(
                    layer_id, look_indexs, X_cur_train.shape, X_cur_test.shape))
                # Fit on X_cur, predict to update X_proba
                y_train_proba_li = np.zeros((X_train.shape[0], n_classes))###训练集的长度 数值为0
                y_test_proba_li = np.zeros((X_test.shape[0], n_classes))###测试集的长度 数值为0
                for ei, est_config in enumerate(self.est_configs):
                    if self.estimators_enlarge:
                        feature_enlarge_ratio = 1.0
                        if layer_id > 0:
                            feature_enlarge_ratio = (n_classes*len(self.est_configs)+X_train.shape[1])/float(X_train.shape[1])
                        ratio_decay = min(self.train_num_level[layer_id]/float(self.train_num_level[0])*feature_enlarge_ratio,1.0)
                        est = self._init_estimators_enlarge(layer_id, ei, ratio_decay)
                        LOGGER.info(
                            "[layer={}] train_decay_level={}, # n_estimators before={}, # n_estimators now={}".format(
                                layer_id, ratio_decay, est_config["n_estimators"],
                                int(est_config["n_estimators"] / ratio_decay)))
                    else:
                        est = self._init_estimators(layer_id, ei)###li 层数 ei 森林id
                    # fit_trainsform
                    test_sets = [("test", X_cur_test, y_test)] if n_tests > 0 else None
                    y_probas = est.fit_transform(X_cur_train, y_train, y_train,
                            test_sets=test_sets, eval_metrics=self.eval_metrics,
                            keep_model_in_mem=train_config.keep_model_in_mem)
                    # train
                    X_proba_train[:, ei * n_classes: ei * n_classes + n_classes] = y_probas[0]
                    y_train_proba_li += y_probas[0]
                    # test
                    if n_tests > 0:
                        X_proba_test[:, ei * n_classes: ei * n_classes + n_classes] = y_probas[1]
                        y_test_proba_li += y_probas[1]
                    if train_config.keep_model_in_mem:
                        self._set_estimator(layer_id, ei, est)
                y_train_proba_li /= len(self.est_configs)
                train_avg_acc2 = calc_accuracy(y_train, np.argmax(y_train_proba_li, axis=1),
                                               'layer_{} - train part2 accuracy'.format(layer_id))
                train_avg_acc = (train_avg_acc2 * y_train.shape[0] + num_train_correct * 100.0) / n_trains
                LOGGER.info('------------------------layer_{} - train accuracy {}'.format(layer_id, train_avg_acc))
                train_acc_list.append(train_avg_acc)
                if n_tests > 0:
                    y_test_proba_li /= len(self.est_configs)
                    test_avg_acc2 = calc_accuracy(y_test, np.argmax(y_test_proba_li, axis=1),
                                                   'layer_{} - test part2 accuracy'.format(layer_id))
                    test_avg_acc = (test_avg_acc2 * y_test.shape[0] + num_test_correct * 100.0) / n_tests
                    LOGGER.info('------------------------layer_{} - test accuracy {}'.format(layer_id, test_avg_acc))
                    test_acc_list.append(test_avg_acc)
                else:
                    test_acc_list.append(0.0)

                opt_layer_id = get_opt_layer_id(test_acc_list if stop_by_test else train_acc_list)
                # set opt_datas
                if opt_layer_id == layer_id:
                    opt_datas = [X_proba_train, y_train, X_proba_test if n_tests > 0 else None, y_test]
                # early stop
                if self.early_stopping_rounds > 0 and layer_id - opt_layer_id >= self.early_stopping_rounds:
                    # log and save final result (opt layer)
                    LOGGER.info("[Result][Optimal Level Detected] opt_layer_num={}, accuracy_train={:.2f}%, accuracy_test={:.2f}%".format(
                        opt_layer_id + 1, train_acc_list[opt_layer_id], test_acc_list[opt_layer_id]))

                    if data_save_dir is not None:
                        self.save_data( data_save_dir, opt_layer_id, *opt_datas)
                    # remove unused model
                    if train_config.keep_model_in_mem:
                        for li in range(opt_layer_id + 1, layer_id + 1):
                            for ei, est_config in enumerate(self.est_configs):
                                self._set_estimator(li, ei, None)
                    self.opt_layer_num = opt_layer_id + 1
                    #return opt_layer_id, opt_datas[0], opt_datas[1], opt_datas[2], opt_datas[3]

                    return test_acc_list[opt_layer_id]

                # save opt data if needed
                if self.data_save_rounds > 0 and (layer_id + 1) % self.data_save_rounds == 0:
                    self.save_data(data_save_dir, layer_id, *opt_datas)
                # inc layer_id
                layer_id += 1
            LOGGER.info("[Result][Reach Max Layer] opt_layer_num={}, accuracy_train={:.2f}%, accuracy_test={:.2f}%".format(
                opt_layer_id + 1, train_acc_list[opt_layer_id], test_acc_list[opt_layer_id]))

            if data_save_dir is not None:
                self.save_data(data_save_dir, self.max_layers - 1, *opt_datas)
            self.opt_layer_num = self.max_layers
            #return self.max_layers, opt_datas[0], opt_datas[1], opt_datas[2], opt_datas[3]
            return test_acc_list[opt_layer_id]
        except KeyboardInterrupt:
            pass

    def transform(self, X_groups_test,y_test=None):
        if not type(X_groups_test) == list:
            X_groups_test = [X_groups_test]
        LOGGER.info("X_groups_test.shape={}".format([xt.shape for xt in X_groups_test]))
        # check look_indexs_cycle
        look_indexs_cycle = self._check_look_indexs_cycle(X_groups_test, False)
        # check group_dims
        group_starts, group_ends, group_dims, X_test = self._check_group_dims(X_groups_test, False)
        LOGGER.info("group_dims={}".format(group_dims))
        LOGGER.info("X_test.shape={}".format(X_test.shape))

        n_tests = X_groups_test[0].shape[0]
        n_classes = self.n_classes

        # probability of each cascades's estimators
        X_proba_test = np.zeros((X_test.shape[0], n_classes * self.n_estimators_1), dtype=np.float32)
        X_cur_test = None

        y_test_proba_li = np.zeros((n_tests, n_classes))
        test_index_now = np.arange(X_test.shape[0])

        for layer_id in range(self.opt_layer_num):
            LOGGER.info("[layer={}], #instances={}".format(layer_id, test_index_now.shape[0]))
            # Copy previous cascades's probability into current X_cur
            if layer_id == 0:
                # first layer not have probability distribution
                X_cur_test = np.zeros((test_index_now.shape[0], 0), dtype=np.float32)
            else:
                X_cur_test = X_proba_test[test_index_now, :].copy()
            # Stack data that current layer needs in to X_cur
            look_indexs = look_indexs_cycle[layer_id % len(look_indexs_cycle)]
            for _i, i in enumerate(look_indexs):
                X_cur_test = np.hstack((X_cur_test, X_test[test_index_now, group_starts[i]:group_ends[i]]))
            LOGGER.info("[layer={}] look_indexs={}, X_cur_test.shape={}".format(
                layer_id, look_indexs, X_cur_test.shape))
            y_test_proba_li[test_index_now,:] = 0
            for ei, est_config in enumerate(self.est_configs):
                est = self._get_estimator(layer_id, ei)
                if est is None:
                    raise ValueError("model (li={}, ei={}) not present, maybe you should set keep_model_in_mem to True".format(
                        layer_id, ei))
                y_probas = est.predict_proba(X_cur_test)
                X_proba_test[test_index_now, ei * n_classes:ei * n_classes + n_classes] = y_probas
                y_test_proba_li[test_index_now, :] += y_probas

            y_test_proba_li[test_index_now, :] /= len(self.est_configs)

            if layer_id < self.opt_layer_num-1:
                CX_test = np.max(y_test_proba_li[test_index_now, :], axis=1)
                test_index_now = test_index_now[CX_test < self.conf_thresh[layer_id]]

            if y_test is not None:
                test_acc = calc_accuracy(y_test, np.argmax(y_test_proba_li, axis=1), 'layer_{} - test accuracy'.format(layer_id))

        return X_proba_test

    def predict_proba(self, X, y=None):
        # n x (n_est*n_classes)
        y_proba = self.transform(X, y)
        # n x n_est x n_classes
        y_proba = y_proba.reshape((y_proba.shape[0], self.n_estimators_1, self.n_classes))
        y_proba = y_proba.mean(axis=1)
        return y_proba

    def save_data(self, data_save_dir, layer_id, X_train, y_train, X_test, y_test):
        for pi, phase in enumerate(["train", "test"]):
            data_path = osp.join(data_save_dir, "layer_{}-{}.pkl".format(layer_id, phase))
            check_dir(data_path)
            data = {"X": X_train, "y": y_train} if pi == 0 else {"X": X_test, "y": y_test}
            LOGGER.info("Saving Data in {} ... X.shape={}, y.shape={}".format(data_path, data["X"].shape, data["y"].shape))
            with open(data_path, "wb") as f:
                pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)