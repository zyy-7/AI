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

//Sigmoid
double Sigmoid(double num) {
	double result = (double)1 / (1 + exp(-num));
	return result;
}

//Sigmoid�����ĵ���
double DerivativeOfSigmoid(double num) {
	double result = (double)num * (1 - num);
	return result;
}

//�����ݽ��г�ʼ��
void Init(){
	Step = 0.05;
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
			fields.push_back(double(1));
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
	for (int i = 0; i < Train[0].size(); i++)
	{
		vector<double> w;
		for (int j = 0; j < cnt_hidden_node - 1; j++)
		{
			double p =((double)rand()) / RAND_MAX;
	//		double p = rand() % 100 / 101 - 0.10000;
			p = 2.00000 * p - 1.00000;
			w.push_back(p);
		}
		Wij.push_back(w);
	}

	for (int i = 0; i < cnt_hidden_node; i++)
	{
		double p = ((double)rand()) / RAND_MAX;
	//	double p = rand() % 100 / (double)101 - 0.10000;
		p = 20.00000 * p - 10.00000;
		Wj.push_back(p);
	}
}

//���ز�����
vector<double> getOutH(int index) {
	vector<double> OutH;

	//h0
	OutH.push_back(Sigmoid((double)1));

	for (int i = 0; i < cnt_hidden_node - 1; i++) {
		double inh = 0;
		for (int j = 0; j < Train[index].size(); j++) {
			inh += Train[index][j] * Wij[j][i];
		}
		OutH.push_back(Sigmoid(inh));
	}

	return OutH;
}

//���������
double getOutY(vector<double> OutH) {
	double InY = 0;
	for (int i = 0; i < OutH.size(); i++) {
		InY += OutH[i] * Wj[i];
	}
	double OutY = Sigmoid(InY);
//	cout << OutY << endl;
	return OutY;
}

//����Ȩ������W
void RefreshW() {
	vector<double> CostWj;
	for (int i = 0; i < Wj.size(); i++) {
		CostWj.push_back((double)0);
	}

	vector<vector<double>> CostWij;
	for (int i = 0; i < Wij.size(); i++) {
		vector<double> costwij;
		for (int j = 0; j < Wij[i].size(); j++) {
			costwij.push_back((double)0);
		}
		CostWij.push_back(costwij);
	}

	srand(time(NULL));
	for (int i = 0; i < cnt_batch; i++) {
		int index = rand() % Train.size();
		vector<double> OutH = getOutH(index);
		double OutY = getOutY(OutH);

		double temp = (OutY - Train_Result[index]) * DerivativeOfSigmoid(OutY) / (double)cnt_batch;
		for (int j = 0; j < Wj.size(); j++) {
			CostWj[j] += temp * OutH[i];
		}

		for (int j = 0; j < Wij.size(); j++) {
			for (int k = 0; k < Wij[j].size(); k++) {
				CostWij[j][k] += temp * Wj[k] * DerivativeOfSigmoid(OutH[k + 1]) * Train[index][j];
			}
		}
	}

	//����Wj
	for (int i = 0; i < Wj.size(); i++) {
		Wj[i] -= Step * CostWj[i];
	}

	//����Wij
	for (int i = 0; i < Wij.size(); i++) {
		for (int j = 0; j < Wij[i].size(); j++) {
			Wij[i][j] -= CostWij[i][j];
		}
	}
}

void getPredictResult() {
	for (int i = 0; i < Test.size(); i++) {
		vector<double> OutH = getOutH(i);
		double y = getOutY(OutH);
		cout << y << endl;
		if (y > 0.5)
			Predict_Result.push_back(1);
		else
			Predict_Result.push_back(0);

	/*	for (int i = 0; i < OutH.size(); i++)
			cout << OutH[i] << endl;*/
	}
}

int main(){
	Init();
	InputTrain();
	InputTest(isTest);
	Train = DealWithData(Train);
	Test = DealWithData(Test);
	InitW();
	
	for (int i = 0; i < Wj.size(); i++) {
		cout << Wj[i] << " ";
	}
	cout << endl;

	for (int i = 0; i < Wij.size(); i++) {
		for (int j = 0; j < Wij[i].size(); j++) {
			cout << Wij[i][j] << " ";
		}
		cout << endl;
	}

	cout << endl << endl;

	//���е���������W
	for (int i = 0; i < cnt_times; i++) {
		RefreshW();
	}

	getPredictResult();

	getOutH(0);

	ofstream outf;
	outf.open("C:/Users/Yuying/Desktop/test_result.txt");
	int cnt_right = 0;
	if (!isTest) {
		for (int i = 0; i < Predict_Result.size(); i++) {
			outf << Predict_Result[i] << endl;
			if (Predict_Result[i] == Validation_Result[i])
				cnt_right++;
		}
	}

	double right = (double)cnt_right / Predict_Result.size();

	cout << right << endl;

	for (int i = 0; i < Wj.size(); i++) {
		cout << Wj[i] << " ";
	}
	cout << endl;

	for (int i = 0; i < Wij.size(); i++) {
		for (int j = 0; j < Wij[i].size(); j++) {
			cout << Wij[i][j] << " ";
		}
		cout << endl;
	}


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