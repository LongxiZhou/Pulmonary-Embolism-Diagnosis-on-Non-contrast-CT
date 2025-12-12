import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import PolynomialFeatures


class WeightedDynamicPEPredictorQuadratic:
    def __init__(self, rer_list, PE_label, weight_new=10, use_quadratic=True):
        """
        初始化系统，支持加权更新，可选择是否包含二次项。

        参数：
            rer_list (list): 已知的 RER 数据列表。
            PE_label (list): 对应的标签列表（1 表示阳性，0 表示阴性）。
            weight_new (int): 新数据的权重倍数。
            use_quadratic (bool): 是否使用二次项特征（log(RER)^2）
        """
        self.rer_list = np.array(rer_list)
        self.PE_label = np.array(PE_label)
        self.weights = np.ones(len(rer_list))  # 初始数据权重为1
        self.weight_new = weight_new  # 新数据的权重倍数
        self.new_rer_labels = []  # 记录新数据的 RER 和标签
        self.incidence_new = 0  # 新数据的阳性比例
        self.incidence_prior = np.mean(PE_label)  # 初始阳性比例
        self.incidence_overall = self.incidence_prior  # 综合阳性比例
        self.log_reg = None  # Logistic Regression 模型
        self.log_reg_original = None  # 保存原始 Logistic Regression 模型
        self.use_quadratic = use_quadratic  # 是否使用二次项
        self.update_model_params(initial=True)

    def _get_log_rer_features(self, rer_values):
        """
        将RER转换为特征，可选择是否包含二次项。

        参数：
            rer_values: RER值或数组

        返回：
            特征数组，如果use_quadratic=True则为(-1, 2)，否则为(-1, 1)
        """
        # 将RER值转换为数组
        rer_array = np.array(rer_values)

        # 处理RER小于2的情况：根据临床意义，RER<2可能是仪器噪声，设为2
        rer_array = np.maximum(rer_array, 2)

        # 计算log(RER)
        log_rer = np.log(rer_array)

        if self.use_quadratic:
            # 如果使用二次项，返回 [log(RER), log(RER)^2]
            log_rer_squared = log_rer ** 2
            return np.column_stack((log_rer, log_rer_squared))
        else:
            # 如果不使用二次项，只返回 log(RER)
            return log_rer.reshape(-1, 1)

    def update_model_params(self, initial=False):
        """根据当前 RER 数据和标签更新模型参数，包括 Logistic Regression。"""
        # 准备数据
        X = self._get_log_rer_features(self.rer_list)
        y = self.PE_label

        # 拟合 Logistic Regression，加入权重信息
        log_reg = LogisticRegression()
        log_reg.fit(X, y, sample_weight=self.weights)

        self.log_reg = log_reg

        # 如果是初始模型，保存为原始模型
        if initial:
            self.log_reg_original = LogisticRegression()
            self.log_reg_original.fit(X, y, sample_weight=self.weights)

    def predict(self, r_new):
        """
        根据 RER 值计算后验概率。

        参数：
            r_new (float): 输入的 RER 值。

        返回：
            float: 后验阳性概率。
        """
        # 使用 Logistic Regression 预测概率
        X_features = self._get_log_rer_features([r_new])
        return self.log_reg.predict_proba(X_features)[0, 1]

    def predict_original(self, r_new):
        """
        使用原始模型预测后验概率。

        参数：
            r_new (float): 输入的 RER 值。

        返回：
            float: 原始模型的后验阳性概率。
        """
        # 使用原始模型预测概率
        X_features = self._get_log_rer_features([r_new])
        return self.log_reg_original.predict_proba(X_features)[0, 1]

    def update_model(self, r_new, label_new):
        """
        根据新的 RER 和标签动态更新模型。

        参数：
            r_new (float): 新的 RER 值。
            label_new (int): 新的标签（1 表示阳性，0 表示阴性）。
        """
        # 添加新数据和权重
        self.rer_list = np.append(self.rer_list, r_new)
        self.PE_label = np.append(self.PE_label, label_new)
        self.weights = np.append(self.weights, self.weight_new)

        # 记录新数据
        self.new_rer_labels.append((r_new, label_new))

        # 更新新数据的阳性比例
        new_labels = np.array([label for _, label in self.new_rer_labels])
        self.incidence_new = np.mean(new_labels)

        # 更新综合阳性比例
        total_labels = np.concatenate([self.PE_label, new_labels])
        self.incidence_overall = np.mean(total_labels)

        # 更新模型参数
        self.update_model_params()

    def plot_rer_posterior_prob(self):
        """
        绘制 RER 值与后验概率的关系。

        横轴：RER（0 到 30）
        纵轴：后验阳性概率（0 到 1）
        增加原始模型和更新后模型的对比曲线。
        """
        # 创建 RER 范围
        rer_values = np.linspace(2, 20, 300)  # 从2开始，避免log(0)
        posterior_probs = [self.predict(r) for r in rer_values]
        posterior_probs_original = [self.predict_original(r) for r in rer_values]

        # 绘图
        plt.figure(figsize=(10, 8))
        plt.plot(rer_values, posterior_probs_original,
                 label='Original Posterior Probability for External Test Set',
                 lw=2, color='blue')
        plt.plot(rer_values, posterior_probs,
                 label='Adjusted Posterior Probability for Prospective Test Set',
                 lw=2, color='red')
        plt.xlabel('Relative Thrombi Ratio (RTR)', fontsize=14)
        plt.ylabel('Posterior PE Probability', fontsize=14)
        plt.title(f'Logistic Regression with {"Quadratic" if self.use_quadratic else "Linear"} Features', fontsize=16)
        plt.grid(True)
        plt.ylim(0, 1)
        plt.xlim(2, 20)
        plt.legend(loc="lower right", fontsize=10)
        plt.show()

    def plot_log_rer_posterior_prob(self):
        """
        绘制 log(RER) 值与后验概率的关系。

        横轴：log(RER)
        纵轴：后验阳性概率（0 到 1）
        增加原始模型和更新后模型的对比曲线。
        """
        # 创建 RER 范围并计算log(RER)
        rer_values = np.linspace(0.001, 30, 300)  # 从0.001开始避免log(0)
        log_rer_values = np.log(np.maximum(rer_values, 2))
        posterior_probs = [self.predict(r) for r in rer_values]
        posterior_probs_original = [self.predict_original(r) for r in rer_values]

        # 绘图
        plt.figure(figsize=(10, 8))
        plt.plot(log_rer_values, posterior_probs_original,
                 label='Original Posterior Probability',
                 lw=2, color='blue')
        plt.plot(log_rer_values, posterior_probs,
                 label='Adjusted Posterior Probability',
                 lw=2, color='red')
        plt.xlabel('log(Relative Thrombi Ratio)')
        plt.ylabel('Posterior PE Probability')
        plt.title(f'Logistic Regression with {"Quadratic" if self.use_quadratic else "Linear"} Features', fontsize=15)
        plt.grid(True)
        plt.ylim(0, 1)
        plt.legend(loc="lower right", fontsize=10)
        plt.show()

    def compute_brier_score(self):
        """
        计算当前记录数据的 Brier Score。

        Brier Score 定义为：
        BS = (1/N) * sum((predicted - actual)^2)
        """
        if not self.new_rer_labels:
            print("No new data to compute Brier Score.")
            return None

        # 提取 RER 和标签
        rer_values, labels = zip(*self.new_rer_labels)
        predicted_probs = [self.predict(r) for r in rer_values]

        # 计算 Brier Score
        brier_score = np.mean([(p - y) ** 2 for p, y in zip(predicted_probs, labels)])
        return brier_score

    def compute_calibration_metrics(self):
        """
        计算校准指标：Expected Calibration Error (ECE) 和 Calibration Slope & Intercept。
        """
        if not self.new_rer_labels:
            print("No new data to compute calibration metrics.")
            return None

        # 提取 RER 和标签
        rer_values, labels = zip(*self.new_rer_labels)
        predicted_probs = np.array([self.predict(r) for r in rer_values])
        labels = np.array(labels)

        # 计算 ECE
        bin_edges = np.linspace(0, 1, 11)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        ece = 0
        for i in range(len(bin_edges) - 1):
            in_bin = (predicted_probs >= bin_edges[i]) & (predicted_probs < bin_edges[i + 1])
            if np.any(in_bin):
                avg_confidence = np.mean(predicted_probs[in_bin])
                avg_accuracy = np.mean(labels[in_bin])
                ece += np.abs(avg_confidence - avg_accuracy) * np.sum(in_bin) / len(predicted_probs)

        # 计算 Calibration Slope 和 Intercept
        reg = LinearRegression().fit(predicted_probs.reshape(-1, 1), labels)
        slope = reg.coef_[0]
        intercept = reg.intercept_

        return ece, slope, intercept

    def plot_calibration_curve(self):
        """
        绘制校准曲线，每个点表示 RER 值的后验概率和真实标签，以及回归线。
        """
        if not self.new_rer_labels:
            print("No new data to plot calibration curve.")
            return

        # 提取 RER 和标签
        rer_values, labels = zip(*self.new_rer_labels)
        predicted_probs = np.array([self.predict(r) for r in rer_values])
        labels = np.array(labels)

        # 绘制散点图
        plt.figure(figsize=(8, 6))
        plt.scatter(predicted_probs, labels, alpha=0.5, label="RTR and PE diagnosis", s=5)

        # 绘制回归线
        reg = LinearRegression().fit(predicted_probs.reshape(-1, 1), labels)
        slope = reg.coef_[0]
        intercept = reg.intercept_
        x_vals = np.linspace(0, 1, 100)
        y_vals = slope * x_vals + intercept
        plt.plot(x_vals, y_vals, color="red", lw=2,
                 label=f"Calibration Line (slope={slope:.4f}, intercept={intercept:.4f})")

        # 绘制参考线
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")

        plt.xlabel("Posterior PE Probability", fontsize=14)
        plt.ylabel("PE Diagnosis (Pos.=1; Neg.=0)", fontsize=14)
        plt.title("Calibration Curve for Prospective Test Set", fontsize=15)
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True)
        plt.show()

    def get_model_coefficients(self):
        """
        获取当前模型的系数。

        返回：
            如果使用二次项：[intercept, coef_log, coef_log_squared]
            如果使用线性项：[intercept, coef_log]
        """
        if self.log_reg is None:
            return None

        intercept = self.log_reg.intercept_[0]
        coefficients = self.log_reg.coef_[0]

        return np.concatenate([[intercept], coefficients])

    def compare_models(self, rer_range=(2, 20), num_points=1000):
        """
        比较线性模型和二次模型的预测结果。

        参数：
            rer_range: RER范围
            num_points: 点数

        返回：
            tuple: (rer_values, linear_probs, quadratic_probs)
        """
        # 创建两个模型：一个线性，一个二次
        linear_model = WeightedDynamicPEPredictorQuadratic(
            self.rer_list, self.PE_label, self.weight_new, use_quadratic=False
        )
        linear_model.log_reg = self.log_reg  # 使用当前模型的线性部分

        quadratic_model = WeightedDynamicPEPredictorQuadratic(
            self.rer_list, self.PE_label, self.weight_new, use_quadratic=True
        )
        quadratic_model.log_reg = self.log_reg  # 使用当前模型的二次部分

        # 计算预测概率
        rer_values = np.linspace(rer_range[0], rer_range[1], num_points)
        linear_probs = [linear_model.predict(r) for r in rer_values]
        quadratic_probs = [quadratic_model.predict(r) for r in rer_values]

        return rer_values, linear_probs, quadratic_probs


if __name__ == '__main__':
    # 示例用法
    import Tool_Functions.Functions as Functions

    # 加载数据
    positives = Functions.pickle_load_object(
        '/Volumes/Longxi_ET1/data_disk/pulmonary_embolism_final/pickle_objects/rer_for_PE_positives.pickle')
    negatives = Functions.pickle_load_object(
        '/Volumes/Longxi_ET1/data_disk/pulmonary_embolism_final/pickle_objects/rer_for_PE_negatives.pickle')

    print("original:", len(positives), len(negatives))


    def clean(data_list):
        new_list = []
        for item in data_list:
            if not 0 < item < np.inf:
                continue
            new_list.append(item)
        return new_list


    positives = clean(positives)
    negatives = clean(negatives)

    rer_prior = positives + negatives
    label_prior = []
    for i in positives:
        label_prior.append(1)
    for j in range(1):
        for i in negatives:
            label_prior.append(0)

    # 使用包含二次项的模型
    posterior_class = WeightedDynamicPEPredictorQuadratic(rer_prior, label_prior, weight_new=10, use_quadratic=True)

    print("original_incidence", posterior_class.incidence_overall)

    # 获取模型系数
    coefficients = posterior_class.get_model_coefficients()
    print(f"Model coefficients: {coefficients}")

    # 加载前瞻性数据
    positive_rer_prospective = Functions.pickle_load_object(
        '/Volumes/Longxi_ET1/data_disk/pulmonary_embolism_final/data_clinical_translation/positive_rer.pickle')
    negative_rer_prospective = Functions.pickle_load_object(
        '/Volumes/Longxi_ET1/data_disk/pulmonary_embolism_final/data_clinical_translation/negative_rer.pickle')

    new_sequence = []
    for value in positive_rer_prospective:
        new_sequence.append((value, 1))
    for value in negative_rer_prospective:
        new_sequence.append((value, 0))

    print("First 10 new samples:", new_sequence[0:10])

    # 更新模型
    for item in new_sequence:
        posterior_class.update_model(item[0], label_new=item[1])

    threshold = 4.5363
    print("RTR at probability 0.5 is 4.5363: P(4.5363) = ", posterior_class.predict(threshold))
    print("Sensitivity:", np.sum(np.array(np.array(positive_rer_prospective) > threshold)) / len(positive_rer_prospective))
    print(np.sum(np.array(np.array(positive_rer_prospective) > threshold)), len(positive_rer_prospective))
    print("Specificity:", np.sum(np.array(np.array(negative_rer_prospective) <= threshold)) / len(negative_rer_prospective))
    print(np.sum(np.array(np.array(negative_rer_prospective) <= threshold)), len(negative_rer_prospective))
    exit()

    # 评估
    print("Brier Score:", posterior_class.compute_brier_score())

    # 可视化
    posterior_class.plot_rer_posterior_prob()

    print("New incidence:", posterior_class.incidence_new)

    # 校准指标
    calibration_metrics = posterior_class.compute_calibration_metrics()
    if calibration_metrics:
        ece, slope, intercept = calibration_metrics
        print(f"Calibration Metrics: ECE={ece:.4f}, slope={slope:.4f}, intercept={intercept:.4f}")

    # 校准曲线
    posterior_class.plot_calibration_curve()
