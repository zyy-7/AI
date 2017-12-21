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
vector<int> Train_Result;
//验证集数据/测试集数据
vector<vector<double>> Test;
//验证集结果
vector<int> Validation_Result;
//预测结果
vector<int> Predict_Result;
//权重向量Wij（输入层到隐藏层）
vector<vector<double>> Wij;
//权重向量Wj（隐藏层到输出层）
vector<double> Wj;
//学习率
double Step;
//隐藏层的节点数（包括偏置）
int cnt_hidden_node;
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

//将字符串转化为浮点数
double stringToNum(string str){
	istringstream iss(str);
	double num;
	iss >> num;
	return num;
}

//激活函数
double Relu(double x)
{
	if (x > 0)
		return x;
	else
		return (double)0;
}

//激活函数的导数
double DerivativeOfRelu(double x)
{
	if (x > 0)
		return (double)1;
	else
		return (double)0;
}

//对数据进行初始化
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

//读入训练集
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

//初始化权重向量（0-1之间）
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

//隐藏层的输入
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