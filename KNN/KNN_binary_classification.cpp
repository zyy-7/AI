#include<iostream>
#include<fstream>
#include<string>
#include<sstream>
#include<vector>
#include<map>
#include<algorithm>
#include <cstdlib>
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
//�趨kֵ 
int K;
//ѵ����·��
string TrainPath;
//��֤��·��
string ValidationPath;
//���Լ�·��
string TestPath;
//�ж��Ƕ�����֤�����ǲ��Լ�
bool isTest;

//���ַ���ת��Ϊ������
double stringToNum(string str) {
	istringstream iss(str);
	double num;
	iss >> num;
	return num;
}

string IntToString(int num) {
	stringstream ss;
	ss << num; 
	string str = ss.str();
	return str;
}


//map��value���� 
bool comp_by_value(pair<int, double> &p1, pair<int, double> &p2) {
    return p1.second < p2.second;
}

struct CompByValue {
    bool operator()(pair<int, double> &p1, pair<int, double> &p2) {
        return p1.second < p2.second;
    }
};

//�����ݽ��г�ʼ��
void Init(){
	K = 1; 
	isTest = 0;
	TrainPath = "C:/Users/Yuying/Desktop/train.csv";
	ValidationPath = "C:/Users/Yuying/Desktop/validation.csv";
	TestPath = "C:/Users/Yuying/Desktop/test.csv";
}

//����ѵ����
void InputTrain(int index, double mNum){
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
		field[index] *= mNum; 
		Train_Result.push_back(fields[fields.size() - 1]);
		fields.pop_back();
		Train.push_back(fields);
	}
}

//������Լ�����֤����
void InputTest(bool isTest, int index, double mNum){
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
		field[index] *= mNum;
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

vector<double> getPredictResult() {
	vector<double> PredictResult;
	for (int i = 0; i < Test.size(); i++) {
	//	cout << i << endl;
		double result = 0;
		map<int, double> distance;
		for (int j = 0; j < Train.size(); j++) {
			double d = 0;
			for (int k = 0; k < Test[i].size(); k++) {
				//d += (Test[i][k] - Train[j][k]) * (Test[i][k] - Train[j][k]);
				//�����پ���
				d += abs(Test[i][k] - Train[j][k]);
			}
			distance[j] = d;
		//	cout << d << endl; 
		}
		vector<pair<int, double>> vec(distance.begin(), distance.end());
    	sort(vec.begin(), vec.end(), comp_by_value);
    	int cnt_k = 0;
    	int cnt_yes = 0;
    	vector<pair<int,double> >::iterator it;
		for(it = vec.begin(); it!= vec.end(); it++){
			cnt_k++;
			if(Train_Result[it->first] != 0)
		    	cnt_yes++;
		    if(cnt_k == K)
		    	break;
		}
	//	cout << cnt_yes << endl;
		if(cnt_yes >= (K - cnt_yes))
			result = 1;
		else
			result = 0;
	//	cout << result << endl;
		PredictResult.push_back(result);
	}
	return PredictResult;
} 

int main(){
	Init();
	//Train = DealWithData(Train);
	//Test = DealWithData(Test);
	
//	int Myindex = 6;
//	double mNum = 500.0;
	
	for(int j = 0; j < 1; j++) {
		if(!Train.empty())
			Train.clear();
		if(!Test.empty())
			Test.clear();
		InputTrain(Myindex, mNum);
		InputTest(isTest, Myindex, mNum);
		Predict_Result = getPredictResult();
		string v = IntToString(j);
		string filename = "predict" + v + ".csv";
		ofstream out(filename);
		for (int i = 0; i < Predict_Result.size(); i++) {
			out << Predict_Result[i] << endl;
		}
		out.close();	
//		mNum += 1.0;
/*		if(j == 3){
			Myindex = 6;
			mNum = 1.5;
		}
		else if(j == 6){
			Myindex = 10;
			mNum = 1.5;
		}*/ 
		
	}
	
	
	
	int cnt_right = 0;
	for (int i = 0; i < Predict_Result.size(); i++) {
		if(Predict_Result[i] == Validation_Result[i])
			cnt_right++;
	}
	
	double right = (double) cnt_right / Predict_Result.size();
	
	cout << right << endl;
	
	return 0;
}
