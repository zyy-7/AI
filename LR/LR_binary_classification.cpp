#include<iostream>
#include<fstream>
#include<string>
#include<sstream>
#include<vector>
#include<map>
#include<algorithm>
#include <cstdlib>
#include <ctime>
#define N 99999 //��λС��
using namespace std;

//ѵ�������ݣ������������
vector<vector<double>> Train;
//ѵ�������
vector<double> Train_Result;
//��֤������/���Լ�����
vector<vector<double>> Test;
//��֤�����
vector<double> Validation_Result;
//Ԥ����
vector<double> Predict_Result;
//����Ȩ��
vector<double> U; 
//Ȩ������ 
vector<double> W;
//��õ�Ȩ������
vector<double> best_W; 
//ѧϰ��
double Step;
//��������
int cnt_times;
//mini-batch
int cnt_batch;
//ѵ����·��
string TrainPath;
//��֤��·��
string ValidationPath;
//���Լ�·��
string TestPath;
//�ж��Ƕ�����֤�����ǲ��Լ�
bool isTest;
//ÿ����ģ�͵�ͶƱȨ 
vector<double> ModelWeight; 
//��ǰѵ���������������ı�� 
vector<int> NowTrain;
//��ģ�͵ĸ���
int cnt_model; 
//��ǰȨ�����õ�����ȷ��
double nowRight;  
//ѵ������Ԥ����
vector<double> TrainPredictResult; 
//��¼������ģ�͵�Ԥ����
vector<vector<double>> modelResult; 

//���ַ���ת��Ϊ������
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

//��ģ�Ͷ�����Ȩ�ص�Ӱ�� 
double TrainS(double e) {
	double result = sqrt((1 - e) / e);
	return result;
} 

//��ģ�͵�ͶƱȨ 
double ModelW(double e) {
	double result = log(sqrt((1 - e) / e));
	return result;
}

//��̬ѧϰ��
void DynamicStep() {
	Step *= 0.99999;
} 

//��̬��
void DynamicBatch() {
	if(cnt_batch < 200)
		cnt_batch *= 2;
} 

//�����ݽ��г�ʼ��
void Init(){
	Step = 1;
	cnt_times = 10000;
	cnt_batch = 1;
	cnt_model = 10;
	nowRight = 0;
	isTest = 0;
	TrainPath = "C:/Users/Yuying/Desktop/train.csv";
	ValidationPath = "C:/Users/Yuying/Desktop/validation.csv";
	TestPath = "C:/Users/Yuying/Desktop/test.csv";
}

//����ѵ����
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

//������Լ�����֤����
void InputTest(bool isTest){
	//isTestΪ1ʱ��������ǲ��Լ�����������֤��
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

//����Ԥ����
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

//��ʼ������Ȩ��
void InitU() {
	double num = (double) 1 / Train.size();
	for(int i = 0; i < Train.size(); i++) {
		U.push_back(num);
	}
} 

//���ݴ����ʸ�������Ȩ�� 
void FreshU(double e) {
	double s = TrainS(e);
	for(int i = 0; i < NowTrain.size(); i++) {
		if(TrainPredictResult[NowTrain[i]] == Train_Result[NowTrain[i]]) {
			U[NowTrain[i]] /= s;
		}
		else {
			U[NowTrain[i]] *= s;
		}
	}
}

//����Ȩ�صõ���ǰ��ѡ�������ı�� 
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

//��ʼ��Ȩ������
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

//����ĳ��������Ȩ�ط���
double GetNewS(int n) {
	double s = 0;
	for (int i = 0; i < W.size(); i++) 
		s += Train[n][i]*W[i];
	
	return s;
}

//�õ����º��Ȩ������W
void GetNewW() {
	vector<double> Cost;
	for (int i = 0; i < W.size(); i++) {
		Cost.push_back(0);
	}
	
	//mini-batch�ݶ��½�
	srand(time(NULL));
	for (int i = 0; i < cnt_batch; i++) {
		int randNum = rand() % Train.size();
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
		double s = GetNewS(NowTrain[randNum]);
		for(int j = 0; j < W.size(); j++) {
			double temp = Sigmoid(s);
			Cost[j] += Train[NowTrain[randNum]][j] * (temp - Train_Result[NowTrain[randNum]]);
		}
	}
	
	for(int i = 0; i < W.size(); i++)
		W[i] -= Step * Cost[i] / double(cnt_batch);
}

//�������õ�WԤ����Լ��Ľ��
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

//������õ�Ȩ�ض�ѵ��������Ԥ�� 
void getPredictTrainResult() {
	if(!TrainPredictResult.empty()) {
		TrainPredictResult.clear(); 
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
	}
}

int main(){
	Init();
	InputTrain();
	InputTest(isTest);
	Train = DealWithData(Train);
	Test = DealWithData(Test);
	InitU();
	
	for(int i = 0; i < cnt_model; i++) {
		cout << i << endl;
		Init();
		InitW();
		GetNowTrain(i);
		
		for(int i = 0; i < cnt_times; i++) {
			adaGetNewW();
			if ((i + 1) % 100 == 0) {
				DynamicBatch();
				DynamicStep();
			}
		
			GetResult();
		}
		
		double r = GetBestResult();
		getPredictTrainResult();
		double wrong = (double) 1 - r;
		double mWeight = ModelW(wrong);
		FreshU(wrong);
		modelResult.push_back(Predict_Result);
		ModelWeight.push_back(mWeight);
	}
	
	if(!Predict_Result.empty())
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
	
	int cnt_right = 0;
	for(int i = 0; i < Test.size(); i++) {
		if(Predict_Result[i] == Validation_Result[i])
			cnt_right++;
	}
	
	double right = (double) cnt_right / Test.size();
	
	cout << right << endl;
	
	return 0;
}
