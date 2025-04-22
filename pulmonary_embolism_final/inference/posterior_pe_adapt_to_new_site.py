
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LinearRegression


class WeightedDynamicPEPredictor:
    def __init__(self, rer_list, PE_label, weight_new=10):
        """
        初始化系统，支持加权更新。

        参数：
            rer_list (list): 已知的 RER 数据列表。
            PE_label (list): 对应的标签列表（1 表示阳性，0 表示阴性）。
            weight_new (int): 新数据的权重倍数。
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
        self.update_model_params(initial=True)

    def update_model_params(self, initial=False):
        """根据当前 RER 数据和标签更新模型参数，包括 Logistic Regression。"""
        # 准备数据
        X = self.rer_list.reshape(-1, 1)
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
        return self.log_reg.predict_proba(np.array([[r_new]]))[0, 1]

    def predict_original(self, r_new):
        """
        使用原始模型预测后验概率。

        参数：
            r_new (float): 输入的 RER 值。

        返回：
            float: 原始模型的后验阳性概率。
        """
        return self.log_reg_original.predict_proba(np.array([[r_new]]))[0, 1]

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
        rer_values = np.linspace(0, 20, 300)
        posterior_probs = [self.predict(r) for r in rer_values]
        posterior_probs_original = [self.predict_original(r) for r in rer_values]

        # 绘图
        # plt.figure(figsize=(10, 6))
        plt.plot(rer_values, posterior_probs_original, label='Original Posterior Probability', lw=2, color='blue')
        plt.plot(rer_values, posterior_probs, label='Adjusted Posterior Probability', lw=2, color='red')
        plt.xlabel('Relative Emboli Ratio (RER)')
        plt.ylabel('Posterior PE Probability')
        plt.title('RER vs Posterior Probability: Original vs Adjusted')
        plt.grid(True)
        plt.ylim(0, 1)
        plt.xlim(0, 20)
        plt.legend(loc="lower right")
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
        # plt.figure(figsize=(10, 6))
        plt.scatter(predicted_probs, labels, alpha=0.5, label="RER and PE diagnosis", s=5)

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

        plt.xlabel("Posterior PE Probability")
        plt.ylabel("PE Diagnosis (Pos.=1; Neg.=0)")
        plt.title("Calibration Curve")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()


if __name__ == '__main__':
    import Tool_Functions.Functions as Functions

    # RER distribution on the original set (suspected PE patients)
    positives = Functions.pickle_load_object(
    '/data_disk/pulmonary_embolism_final/pickle_objects/rer_for_PE_positives.pickle')
    negatives = Functions.pickle_load_object(
                             '/data_disk/pulmonary_embolism_final/pickle_objects/rer_for_PE_negatives.pickle')

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

    posterior_class = WeightedDynamicPEPredictor(rer_prior, label_prior, weight_new=10)
    print("prob rer 11.06 original:", posterior_class.predict(11.06))

    print("original_incidence", posterior_class.incidence_overall)

    # replace with your own data.
    positive_rer_prospective = Functions.pickle_load_object(
        '/data_disk/pulmonary_embolism_final/data_clinical_translation/positive_rer.pickle')
    negative_rer_prospective = Functions.pickle_load_object(
        '/data_disk/pulmonary_embolism_final/data_clinical_translation/negative_rer.pickle')

    new_sequence = []
    for value in positive_rer_prospective:
        new_sequence.append((value, 1))
    for value in negative_rer_prospective:
        new_sequence.append((value, 0))

    for item in new_sequence:
        posterior_class.update_model(item[0], label_new=item[1])

    print("prob rer 11.06 adjusted:", posterior_class.predict(11.06))

    print("brier", posterior_class.compute_brier_score())
    posterior_class.plot_rer_posterior_prob()
    print(posterior_class.incidence_new)

    print(posterior_class.compute_calibration_metrics())
    posterior_class.plot_calibration_curve()
