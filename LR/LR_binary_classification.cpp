#include<iostream>
#include<fstream>
#include<string>
#include<sstream>
#include<vector>
#include<map>
#include<algorithm>
#include <cstdlib>
#include <ctime>
#define N 99999 //五位小数
using namespace std;

//训练集数据（不包括结果）
vector<vector<double>> Train;
//训练集结果
vector<double> Train_Result;
//验证集数据/测试集数据
vector<vector<double>> Test;
//验证集结果
vector<double> Validation_Result;
//预测结果
vector<double> Predict_Result;
//样本权重
vector<double> U; 
//权重向量 
vector<double> W;
//最好的权重向量
vector<double> best_W; 
//学习率
double Step;
//迭代次数
int cnt_times;
//mini-batch
int cnt_batch;
//训练集路径
string TrainPath;
//验证集路径
string ValidationPath;
//测试集路径
string TestPath;
//判断是读入验证集还是测试集
bool isTest;
//每个子模型的投票权 
vector<double> ModelWeight; 
//当前训练集中所有样本的编号 
vector<int> NowTrain;
//子模型的个数
int cnt_model; 
//当前权重所得到的正确率
double nowRight;  
//训练集的预测结果
vector<double> TrainPredictResult; 
//记录所有子模型的预测结果
vector<vector<double>> modelResult; 
//记录当前子模型对每个训练样本的预测概率
vector<double> Trainp; 

//将字符串转化为浮点数
double stringToNum(string str){
	istringstream iss(str);
	double num;
	iss >> num;
	return num;
}

//Sigmoid
double Sigmoid(double num) {
	double result = (double)1 / (1 + exp(-num));
	return result;
}

//子模型对样本权重的影响 
double TrainS(double e) {
	double result = sqrt((1 - e) / e);
	return result;
} 

//子模型的投票权 
double ModelW(double e) {
	double result = log(sqrt((1 - e) / e));
	return result;
}

//动态学习率
void DynamicStep() {
	Step *= 0.99999;
} 

//动态批
void DynamicBatch() {
	if(cnt_batch < 200)
		cnt_batch *= 2;
} 

//对数据进行初始化
void Init(){
	Step = 1;
	cnt_times = 10000;
	cnt_batch = 1;
	cnt_model = 1;
	nowRight = 0;
	isTest = 0;
	TrainPath = "C:/Users/Yuying/Desktop/train.csv";
	ValidationPath = "C:/Users/Yuying/Desktop/validation.csv";
	TestPath = "C:/Users/Yuying/Desktop/test.csv";
}

//读入训练集
void InputTrain(){
	ifstream f(TrainPath);
	string line;
	while (getline(f, line)){
		istringstream sin(line);
		string field;
		vector<double> fields;
		fields.push_back(double(1));
		int cnt = 0;
		while (getline(sin, field, ',')) {
			if (cnt == 1) {
				double num;
				if (field[field.length() - 1] == 'A')
					num = 0;
				else if (field[field.length() - 1] == 'B')
					num = 1;
				else if (field[field.length() - 1] == 'C')
					num = 2;
				else
					num = 3;
				fields.push_back(num);
			}
			else {
				fields.push_back(stringToNum(field));
			}
			cnt++;
		}	
		Train_Result.push_back(fields[fields.size() - 1]);
		fields.pop_back();
		Train.push_back(fields);
	}
}

//读入测试集（验证集）
void InputTest(bool isTest){
	//isTest为1时，读入的是测试集，否则是验证集
	string filePath;
	if (isTest)
		filePath = TestPath;
	else
		filePath = ValidationPath;
	ifstream f(filePath);

	string line;
	while (getline(f, line)) {
		istringstream sin(line);
		string field;
		vector<double> fields;
		fields.push_back(double(1));
		int cnt = 0;
		while (getline(sin, field, ',')) {
			if (cnt == 1) {
				double num;
				if (field[field.length() - 1] == 'A')
					num = 0;
				else if (field[field.length() - 1] == 'B')
					num = 1;
				else if (field[field.length() - 1] == 'C')
					num = 2;
				else
					num = 3;
				fields.push_back(num);
			}
			else {
				fields.push_back(stringToNum(field));
			}
			cnt++;
		}
		if (!isTest) {
			Validation_Result.push_back(fields[fields.size() - 1]);
			fields.pop_back();
			Test.push_back(fields);
		}
		else {
			Test.push_back(fields);
		}
	}
}

//数据预处理
vector<vector<double>> DealWithData(vector<vector<double>> DataSet){
	for (int i = 0; i < DataSet[0].size(); i++) {
		double max_num = 0;
		double min_num = 1000000;
		for (int j = 0; j < DataSet.size(); j++) {
			if (DataSet[j][i] > max_num)
				max_num = DataSet[j][i];
			if (DataSet[j][i] < min_num)
				min_num = DataSet[j][i];
		}
		if (max_num != min_num)
		{
			double denominator = max_num - min_num;
			for (int k = 0; k < DataSet.size(); k++) {
				double numerator = DataSet[k][i] - min_num;
				DataSet[k][i] = numerator / denominator;
			}
		}
		else if (max_num == min_num && max_num != 0) {
			for (int k = 0; k < DataSet.size(); k++)
				DataSet[k][i] /= max_num;
		}
	}
	return DataSet;
}

//初始化样本权重
void InitU() {
	if(!U.empty())
		U.clear();
	double num = (double) 1 / Train.size();
	for(int i = 0; i < Train.size(); i++) {
		U.push_back(num);
	}
} 

//根据错误率更新样本权重 
void FreshU(double e) {
	double s = TrainS(e);
	for(int i = 0; i < NowTrain.size(); i++) {
		if(TrainPredictResult[NowTrain[i]] == Train_Result[NowTrain[i]]) {
			U[NowTrain[i]] /= s;
		}
		else {
			U[NowTrain[i]] *= s;
		}
	/*	if(Train_Result[NowTrain[i]] == 0) {
			if(Trainp[NowTrain[i]] < 0.5 && Trainp[NowTrain[i]] >= 0.45)
				U[NowTrain[i]] *= s;
			else
				U[NowTrain[i]] /= s;
		}
		else if(Train_Result[NowTrain[i]] == 1) {
			if(Trainp[NowTrain[i]] >= 0.5 && Trainp[NowTrain[i]] <= 0.55)
				U[NowTrain[i]] *= s;
			else
				U[NowTrain[i]] /= s;
		}*/
	}
}

//根据权重得到当前被选中样本的编号 
void GetNowTrain(int index) {
	if(!NowTrain.empty()) 
		NowTrain.clear();
	double uSum = 0;
	for(int i = 0; i < U.size(); i++) {
		uSum += U[i];
	}
	
	double uAvg = (double) uSum / U.size();
	for(int i = 0; i < U.size(); i++) {
		if(U[i] > uAvg) {
			NowTrain.push_back(i);
		}
		else if(index == 0) {
			NowTrain.push_back(i);
		}
	}
}

//初始化权重向量
void InitW() {
//	srand(time(NULL));
	if(!W.empty())
		W.clear();
	if(!best_W.empty()) {
		best_W.clear();
	}
	
	for (int i = 0; i < Train[0].size(); i++) {
		double p = 1;
	//	double p = ((double)rand()) / RAND_MAX;
	//	double p = rand() % 100 / (double)101 - 0.10000;
	//	p = 2.00000 * p - 1.00000;
		W.push_back(p);
		best_W.push_back(p);
	}
}

//计算某个样例的权重分数
double GetNewS(int n) {
	double s = 0;
	for (int i = 0; i < W.size(); i++) 
		s += Train[n][i]*W[i];
	
	return s;
}

//得到更新后的权重向量W
void GetNewW() {
	vector<double> Cost;
	for (int i = 0; i < W.size(); i++) {
		Cost.push_back(0);
	}
	
	//mini-batch梯度下降
	srand(time(NULL));
	for (int i = 0; i < cnt_batch; i++) {
		int randNum = rand() % Train.size();
	//	cout << randNum << endl;
		double s = GetNewS(randNum);
		for(int j = 0; j < W.size(); j++) {
			double temp = Sigmoid(s);
			Cost[j] += Train[randNum][j] * (temp - Train_Result[randNum]);
		}
	}
	
	for(int i = 0; i < W.size(); i++)
		W[i] -= Step * Cost[i] / double(cnt_batch);
}

void adaGetNewW() {
	vector<double> Cost;
	for (int i = 0; i < W.size(); i++) {
		Cost.push_back(0);
	}
	
	srand(time(NULL));
	for (int i = 0; i < cnt_batch; i++) {
		int randNum = rand() % NowTrain.size();
	//	cout << randNum << endl;
		double s = GetNewS(NowTrain[randNum]);
		for(int j = 0; j < W.size(); j++) {
			double temp = Sigmoid(s);
			Cost[j] += Train[NowTrain[randNum]][j] * (temp - Train_Result[NowTrain[randNum]]);
		}
	}
	
	for(int i = 0; i < W.size(); i++)
		W[i] -= Step * Cost[i] / double(cnt_batch);
}

//根据所得的W预测测试集的结果
void GetResult() {
	if(!Predict_Result.empty())
		Predict_Result.clear();
	for(int i = 0; i < Test.size(); i++) {
		double s = 0;
		for(int j = 0; j < W.size(); j++)
			s += Test[i][j] * W[j];
		double p = Sigmoid(s);
		
	//	cout << p << endl;
		if(p >= 0.5)
			Predict_Result.push_back(1);
		else
			Predict_Result.push_back(0);
	}
	
	int cnt_right = 0;
	for(int i = 0; i < Test.size(); i++) {
		if(Predict_Result[i] == Validation_Result[i])
			cnt_right++;
	}
	
	double right = (double) cnt_right / Test.size();
	
//	cout << right << endl;
	
	if(right > nowRight) {
		nowRight = right;
		for(int i = 0; i < W.size(); i++) {
			best_W[i] = W[i];
		}
	}
}

double GetBestResult() {
	if(!Predict_Result.empty())
		Predict_Result.clear();
	for(int i = 0; i < Test.size(); i++) {
		double s = 0;
		for(int j = 0; j < best_W.size(); j++)
			s += Test[i][j] * best_W[j];
		double p = Sigmoid(s);
		
	//	cout << p << endl;
		if(p >= 0.5)
			Predict_Result.push_back(1);
		else
			Predict_Result.push_back(0);
	}
	
	int cnt_right = 0;
	for(int i = 0; i < Test.size(); i++) {
		if(Predict_Result[i] == Validation_Result[i])
			cnt_right++;
	}
	
	double right = (double) cnt_right / Test.size();
	
	return right;
}

//根据求得的权重对训练集进行预测 
void getPredictTrainResult() {
	if(!TrainPredictResult.empty()) {
		TrainPredictResult.clear(); 
	}
	
	if(!Trainp.empty()) {
		Trainp.clear();
	}
	
	for(int i = 0; i < Train.size(); i++) {
		double s = 0;
		for(int j = 0; j < best_W.size(); j++)
			s += Train[i][j] * best_W[j];
		double p = Sigmoid(s);
		
	//	cout << p << endl;
		if(p >= 0.5)
			TrainPredictResult.push_back(1);
		else
			TrainPredictResult.push_back(0);
			
		Trainp.push_back(p);
	}
}

int main(){
	Init();
	InputTrain();
	InputTest(isTest);
	Train = DealWithData(Train);
	Test = DealWithData(Test);
	InitU();
	
	srand(time(NULL));
	
	vector<double> f1;
	
	for(int i = 0; i < 50; i++){
		InitU();
		for(int i = 0; i < cnt_model; i++) {
		//	cout << i << endl;
			Init();
			InitW();
			GetNowTrain(i);
			
		//	srand(time(NULL));
			for(int i = 0; i < cnt_times; i++) {
			//	srand(time(NULL));
				adaGetNewW();
				
				if ((i + 1) % 100 == 0) {
					DynamicBatch();
					DynamicStep();
				}
			
				GetResult();
			}
			
			GetBestResult();
			
			int TP = 0;
			int FN = 0;
			int TN = 0;
			int FP = 0;
			
			for(int i = 0; i < Test.size(); i++) {
				if(Predict_Result[i] == Validation_Result[i] && Predict_Result[i] == 1)
					TP++;
				else if(Predict_Result[i] == 0 && Validation_Result[i] == 1)
					FN++;
				else if(Predict_Result[i] == Validation_Result[i] && Predict_Result[i] == 0)
					TN++;
				else if(Predict_Result[i] == 1 && Validation_Result[i] == 0)
					FP++;
			}
		
			double Recall = (double) TP / (TP + FN);
			double Precision = (double) TP / (TP + FP);
			double F1 = (double) 2 * Precision * Recall / (Precision + Recall);
			f1.push_back(F1);
			string filename = "RandomsingleLR.csv";
			ofstream out(filename);
			for (int i = 0; i < f1.size(); i++) {
				out << f1[i] << endl;
			}
			out.close();
		//	cout << F1 << endl; 
			
		/*	cout << endl << endl;
			
			double r = GetBestResult();
			getPredictTrainResult();
			double wrong = (double) 1 - r;
			double mWeight = ModelW(wrong);
			FreshU(wrong);
			modelResult.push_back(Predict_Result);
			ModelWeight.push_back(mWeight);*/ 
		}
		
	/*	if(!Predict_Result.empty())
			Predict_Result.clear();
		
		for(int i = 0; i < Test.size(); i++) {
			double isOne = 0;
			double isZero = 0;
			for(int j = 0; j < cnt_model; j++) {
				if(modelResult[j][i] == 0){
					isZero += ModelWeight[j];
				}
				else{
					isOne += ModelWeight[j];
				}
			}
			if(isZero > isOne){
				Predict_Result.push_back(0);
			}
			else{
				Predict_Result.push_back(1);
			}
		}
		
		int TP = 0;
		int FN = 0;
		int TN = 0;
		int FP = 0;
		
		for(int i = 0; i < Test.size(); i++) {
			if(Predict_Result[i] == Validation_Result[i] && Predict_Result[i] == 1)
				TP++;
			else if(Predict_Result[i] == 0 && Validation_Result[i] == 1)
				FN++;
			else if(Predict_Result[i] == Validation_Result[i] && Predict_Result[i] == 0)
				TN++;
			else if(Predict_Result[i] == 1 && Validation_Result[i] == 0)
				FP++;
		}
	
		double Recall = (double) TP / (TP + FN);
		double Precision = (double) TP / (TP + FP);
		double F1 = (double) 2 * Precision * Recall / (Precision + Recall);
		f1.push_back(F1);*/ 
	//	cout << F1 << endl;
	//	cout << Predict_Result.size() << endl;
	}
	
/*	string filename = "RandomsingleLR.csv";
	ofstream out(filename);
	for (int i = 0; i < f1.size(); i++) {
		out << f1[i] << endl;
	}
	out.close();*/
	
	return 0;
}
