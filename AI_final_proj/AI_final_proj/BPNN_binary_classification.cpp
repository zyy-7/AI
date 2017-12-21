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
vector<int> Train_Result;
//��֤������/���Լ�����
vector<vector<double>> Test;
//��֤�����
vector<int> Validation_Result;
//Ԥ����
vector<int> Predict_Result;
//Ȩ������Wij������㵽���ز㣩
vector<vector<double>> Wij;
//Ȩ������Wj�����ز㵽����㣩
vector<double> Wj;
//ѧϰ��
double Step;
//���ز�Ľڵ���������ƫ�ã�
int cnt_hidden_node;
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

//�����
double Relu(double x)
{
	if (x > 0)
		return x;
	else
		return (double)0;
}

//������ĵ���
double DerivativeOfRelu(double x)
{
	if (x > 0)
		return (double)1;
	else
		return (double)0;
}

//�����ݽ��г�ʼ��
void Init(){
	Step = 0.00001;
	cnt_hidden_node = 20;
	cnt_times = 1000;
	cnt_batch = 1;
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
		Train_Result.push_back(int(fields[fields.size() - 1]));
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
			Validation_Result.push_back(int(fields[fields.size() - 1]));
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

//��ʼ��Ȩ��������0-1֮�䣩
void InitW() {
	srand(time(NULL));
	for (int i = 0; i < Train[0].size(); i++)
	{
		vector<double> w;
		for (int j = 0; j < cnt_hidden_node - 1; j++)
		{
			double p = rand() % (N + 1) / (double)(N + 1);
			w.push_back(p);
		}
		Wij.push_back(w);
	}

	for (int i = 0; i < cnt_hidden_node; i++)
	{
		double p = rand() % (N + 1) / (double)(N + 1);
		Wj.push_back(p);
	}
}

//���ز������
vector<double> InH(int index) {
	
}

int main(){
	Init();
	InputTrain();
	InputTest(isTest);
	Train = DealWithData(Train);
	Test = DealWithData(Test);
	InitW();

	/*for (int i = 0; i < Wj.size(); i++)
		cout << Wj[i] << " ";
	cout << endl;*/

	/*for (int i = 0; i < 100; i++) {
		for (int j = 0; j < Train[i].size(); j++)
			cout << Train[i][j] << " ";
		cout << endl;
	}*/

	/*for (int i = 0; i < 100; i++) {
		for (int j = 0; j < Test[i].size(); j++)
			cout << Test[i][j] << " ";
		cout << endl;
	}*/

	system("pause");
	return 0;
}