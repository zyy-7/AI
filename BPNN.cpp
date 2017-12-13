#include<iostream>
#include<fstream>
#include<string>
#include<sstream>
#include<vector>
#include<map>
#include<algorithm>
#include <cstdlib>
#include <ctime>
#define N 999 //三位小数
using namespace std;

vector<string> tag;
vector<string> date;
vector<vector<double>> Wij;
vector<double> Wj;
vector<vector<double>> X;
//训练集的真实结果 
vector<double> Y;
vector<vector<double>> V_X;
vector<double> V_Y;
vector<vector<double>> Train;
vector<vector<double>> Validation;
double Step;
//层数
int cnt_layer; 
//迭代次数
int cnt_times; 
//隐藏层节点个数 
int cnt_hidden_node; 
//批
int cnt_batch; 
double relu_a = 0.000001;

double stringToNum(string str)  
{  
    istringstream iss(str);  
    double num;  
    iss >> num;  
    return num;      
}  

/*double Sigmoid(double x)
{
	double result = (double) 1 / (1 + exp(-x));
	return result;
}

double DerivativeOfSig(double x)
{
	double result = (double) x*(1 - x);
	return result;	
}*/

double Relu(double x)
{
	if(x > 0)
		return x;
	else
	{
		x *= relu_a;
	}
	return x;
} 

double DerivativeOfRelu(double x)
{
	if(x > 0)
		return (double)1;
	else
	 	return relu_a;
}

void DynamicStep()
{
/*	if(Step > 0.1)
		Step *= 0.09;
	else if(Step > 0.01)
		Step *= 0.5;
	else */
		Step *= 0.99999;
} 

void DynamicBatch()
{
	if(cnt_batch < 100)
		cnt_batch *= 2;
}

vector<vector<double>> Input(string filePath)
{
	vector<vector<double>> DataSet;
	ifstream f(filePath);   
    string line; 
	bool flag = 0;
	
	while(getline(f, line))     
    {
        istringstream sin(line);   
        string field;  
        vector<double> fields;
		if(flag == 0)
		{
			while(getline(sin, field, ','))  
        		tag.push_back(field);
        	flag = 1;
		}   
		else
		{
			int cnt = 0;
			while(getline(sin, field, ',')) 
			{
				if(cnt < 2)
				{
					cnt++;
					if(cnt > 1)
					{
						string temp_str = "";
						if(field[field.length() - 2] != '/')
							temp_str += field[field.length() - 2] + field[field.length() - 1];
						else
							temp_str += field[field.length() - 1];
						fields.push_back(stringToNum(temp_str));
					}
				//		date.push_back(field);
				}
				else
					fields.push_back(stringToNum(field));
				
			}     
        	DataSet.push_back(fields);
		} 
	}
	return DataSet;
//	cout << DataSet.size() << endl;
}

vector<double> GetInputY(vector<vector<double>> DataSet)
{
	vector<double> y;
	for(int i = 0; i < DataSet.size(); i++)
	{
		y.push_back(DataSet[i][DataSet[0].size()-1]);
	}
	return y;
}

void Init()
{
	cnt_layer = 2;
	cnt_times = 10000;
	cnt_hidden_node = 20;
//	Step = 0.7;
	Step = 0.001;
	cnt_batch = 1;
	
	for(int i = 0; i < Train.size(); i++)
	{
		vector<double> x;
		for(int j = 0; j < Train[i].size(); j++)
		{
			if(j == 0)
				x.push_back(1);
			else
				x.push_back(Train[i][j - 1]);
		}
		X.push_back(x);
	}
	
//	cout << Train.size() << endl;
	
	for(int i = 0; i < Validation.size(); i++)
	{
		vector<double> v;
		for(int j = 0; j < Validation[i].size(); j++)
		{
			if(j == 0)
				v.push_back(1);
			else
				v.push_back(Validation[i][j - 1]);
		}
		V_X.push_back(v);
	}
	
	srand(time(NULL));
	for(int i = 0; i < X[0].size(); i++)
	{
		vector<double> w;
		for(int j = 0; j < cnt_hidden_node - 1; j++)
		{
			double p = rand()%(N+1)/(double)(N+1);
			p = (double) p*2 - 1;
			w.push_back(p);
	//		cout << p << " ";
		}
//		cout << endl;
		Wij.push_back(w);
	}	
//	cout << endl;
	
	for(int i = 0; i < cnt_hidden_node; i++)
	{
		double p = rand()%(N+1)/(double)(N+1);
		p = (double) p*2 - 1;
		Wj.push_back(p);
	}
}

//数据预处理
vector<vector<double>> DealWithData(vector<vector<double>> DataSet)
{
	for(int i = 0; i < DataSet[0].size(); i++)
	{
		double min_num = 1000000;
		double max_num = DataSet[0][i];
		for(int j = 0; j < DataSet.size(); j++)
		{
			if(DataSet[j][i] > max_num)
				max_num = DataSet[j][i];
			else if(DataSet[j][i] < min_num)
				min_num = DataSet[j][i];
		}
		if((max_num - min_num) != 0)
		{
			for(int j = 0; j < DataSet.size(); j++)
			{		
				DataSet[j][i] = (DataSet[j][i] - min_num) / (max_num - min_num);
			}
		}
	}
	return DataSet;
} 

vector<double> GetInH(int n, vector<vector<double>> DataSet)
{
	vector<double> InH;
	
	//h0
	InH.push_back(1);
	
	for(int i = 0; i < cnt_hidden_node - 1; i++)
	{
		double temp = 0;
		for(int k = 0; k < DataSet[n].size(); k++)
		{
			temp += DataSet[n][k] * Wij[k][i];
		//	cout << "inh" << X[n][k] * Wij[k][i] << endl;
		}
	//	cout << "InH " << temp << endl;
		InH.push_back(temp);
	}
	return InH;
}

vector<double> GetOutH(vector<double> InH)
{
	vector<double> OutH;
	for(int i = 0; i < InH.size(); i++)
	{
		OutH.push_back(Relu(InH[i]));
	//	OutH.push_back(Sigmoid(InH[i]));
	/*	cout << "In " <<  InH[i] << endl;*/
	//	cout << "Out " << Sigmoid(InH[i]) << endl; 
	}
//	cout << endl;
	return OutH;
}

double GetPreY(vector<double> OutH)
{
	double y = 0;
	for(int j = 0; j < OutH.size(); j++)
	{
		y += Wj[j] * OutH[j];
	//	cout << "~ " << Wj[j] * OutH[j] << endl;
	}
//	cout << "Y:" << y << endl;
	return y;
}

vector<double> GetAllY(vector<vector<double>> DataSet)
{
	vector<double> preY;
	for(int i = 0; i < DataSet.size(); i++)
	{
		vector<double> InH = GetInH(i, DataSet);
		vector<double> OutH = GetOutH(InH);
		double y = GetPreY(OutH);
		preY.push_back(y);
	}
	return preY;
}

double GetA0(int n, double y)
{
	double a0 = Y[n] - y;
/*	cout << a0 << endl;
	cout << endl;*/
	return a0;
}  

vector<double> GetAj(int n, double a0, vector<double> OutH)
{
	vector<double> aj;
	for(int j = 1; j < OutH.size(); j++)
	{
		double a = a0 * Wj[j] * DerivativeOfRelu(OutH[j]);
	//	double a = a0 * Wj[j] * DerivativeOfSig(OutH[j]);
		aj.push_back(a);
	/*	cout << a << endl;
		cout << a0 << endl;
		cout << Wj[j] << endl;
		cout << OutH[j] << endl;
		cout << DerivativeOfSig(OutH[j]) << endl;
		cout << endl;*/ 
	}
	return aj;
}

vector<vector<double>> GetCostOfWij()
{
	vector<vector<double>> CostOfWij;
	for(int i = 0; i < Wij.size(); i++)
	{
		vector<double> cost;
		for(int j = 0; j < Wij[i].size(); j++)
		{
			cost.push_back(0);
		}
		CostOfWij.push_back(cost);
	}
	
	srand(time(NULL));
    for(int i = 0; i < cnt_batch; i++)
    {
    	int n = rand()%Y.size();
    	vector<double> InH = GetInH(n, X);
		vector<double> OutH = GetOutH(InH);
		double y = GetPreY(OutH);
		double a0 = GetA0(n, y);
		vector<double> aj = GetAj(n, a0, OutH);
	//	cout << "hh" << endl;
		for(int i = 0; i < CostOfWij.size(); i++)
	 	{
	 		for(int j = 0; j < CostOfWij[i].size(); j++)
	 		{
	 			CostOfWij[i][j] += aj[j] * X[n][i];
	 //			cout << CostOfWij[i][j] << " ";
	 		}
	 //		cout << endl;
	 	}
    }
	
	return CostOfWij;
}

vector<double> GetCostOfWj()
{
	vector<double> CostOfWj;
	for(int i = 0; i < Wj.size(); i++)
	{
		CostOfWj.push_back(0);
	}
	
	srand(time(NULL));
    for(int i = 0; i < cnt_batch; i++)
    {
    	int n = rand()%Y.size();
    	vector<double> InH = GetInH(n, X);
		vector<double> OutH = GetOutH(InH);
		double y = GetPreY(OutH);
		double a0 = GetA0(n, y);
		for(int i = 0; i < CostOfWj.size(); i++)
		{ 	
			CostOfWj[i] += a0 * OutH[i];
		}
	}
	
	return CostOfWj;
}

//更新权重
void GetNewW(vector<vector<double>> CostOfWij, vector<double> CostOfWj)
{
	for(int i = 0; i < CostOfWj.size(); i++)
	{
	/*	cout << CostOfWj[i] << endl;
		cout << Step << endl;
		cout << cnt_batch << endl;
		cout << Step*CostOfWj[i]/(double)cnt_batch << endl << endl;*/
		Wj[i] += Step*CostOfWj[i]/(double)cnt_batch;
	} 
	
	for(int i = 0; i < CostOfWij.size(); i++)
	{
		for(int j = 0; j < CostOfWij[i].size(); j++)
		{
			Wij[i][j] += Step*CostOfWij[i][j]/(double)cnt_batch;
		}
	}
} 

double MSE(vector<vector<double>> DataSet, vector<double> DataY)
{
	double result = 0;
	vector<double> pre_Y = GetAllY(DataSet);
	for(int i = 0; i < pre_Y.size(); i++)
	{
		result += (pre_Y[i] - DataY[i]) * (pre_Y[i] - DataY[i]);
	//	cout << pre_Y[i] << endl;
	}
	result = (double) result / Y.size();
	return result;
}

int main()
{
	string trainPath = "C:/Users/Yuying/Desktop/train.csv";
	string validationPath = "C:/Users/Yuying/Desktop/test.csv";
	Train = Input(trainPath);
	Validation = Input(validationPath);
	Y = GetInputY(Train);
//	V_Y = GetInputY(Validation);
	Init();
	X = DealWithData(X);
	V_X = DealWithData(V_X);
	
//	cout << V_X.size() << endl;
	
//	cout << V_X.size() << endl;
	
	vector<double> getMse;
	while(cnt_times--)
	{
		vector<double> CostOfWj = GetCostOfWj();
		vector<vector<double>> CostOfWij = GetCostOfWij();
		GetNewW(CostOfWij, CostOfWj);
	//	double mse = MSE(V_X, V_Y);
	//	getMse.push_back(mse); 
		DynamicStep();
		DynamicBatch();
	}
	
	cout << MSE(X, Y) << endl;
	
/*	ofstream outf; 
	outf.open("C:/Users/Yuying/Desktop/V_MSE_Sig.txt");
	for(int i = 0; i < getMse.size(); i++)
	{
		outf << getMse[i] << endl;
	}*/
	
	vector<double> MyY = GetAllY(V_X);
	ofstream outf_;
	outf_.open("C:/Users/Yuying/Desktop/Prediction.txt");
	for(int i = 0; i < MyY.size(); i++)
		outf_ << MyY[i] << endl;
	
/*	for(int i = 0; i < Wj.size(); i++)
		cout << Wj[i] << " ";
	cout << endl;
	
	for(int i = 0; i < Wij.size(); i++)
	{
		for(int j = 0; j < Wij[i].size(); j++)
			cout << Wij[i][j] << " ";
		cout << endl;
	}*/
	
	
/*	for(int i = 0; i < date.size(); i++)
	{
		cout << date[i] << endl;
	}*/
	
}
