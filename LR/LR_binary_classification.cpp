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
//Ȩ������ 
vector<double> W;
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

//�����ݽ��г�ʼ��
void Init(){
	Step = 0.0001;
	cnt_times = 10000;
	cnt_batch = 10;
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

//��ʼ��Ȩ������
void InitW() {
	srand(time(NULL));

	for (int i = 0; i < Train[0].size(); i++) {
		double p = ((double)rand()) / RAND_MAX;
	//	double p = rand() % 100 / (double)101 - 0.10000;
		p = 2.00000 * p - 1.00000;
		W.push_back(p);
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

//�������õ�WԤ����Լ��Ľ��
void GetResult() {
	for(int i = 0; i < Test.size(); i++) {
		double s = 0;
		for(int j = 0; j < W.size(); j++)
			s += Test[i][j] * W[j];
		double p = Sigmoid(s);
		
		cout << p << endl;
		if(p >= 0.5)
			Predict_Result.push_back(1);
		else
			Predict_Result.push_back(0);
	}
}

int main(){
	Init();
	InputTrain();
	InputTest(isTest);
	Train = DealWithData(Train);
	Test = DealWithData(Test);
	InitW();
	
//	GetResult();
	
	for(int i = 0; i < cnt_times; i++) {
		GetNewW();
	}
	
	GetResult();
	
	int cnt_right = 0;
	for(int i = 0; i < Test.size(); i++) {
		if(Predict_Result[i] == Validation_Result[i])
			cnt_right++;
	}
	
	double right = (double) cnt_right / Test.size();
	
	cout << right << endl;
	
	return 0;
}
