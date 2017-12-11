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
vector<vector<double>> Train;
double Step;
//层数
int cnt_layer; 
//迭代次数
int cnt_times; 
//隐藏层节点个数 
int cnt_hidden_node; 

double stringToNum(string str)  
{  
    istringstream iss(str);  
    double num;  
    iss >> num;  
    return num;      
}  

double Sigmoid(double x)
{
	double result = (double) 1 / (1 + exp(-x));
	return result;
}

double DerivativeOfSig(double x)
{
	double result = (double) Sigmoid(x) * (1 - Sigmoid(x));
	return result;	
}

void Input()
{
	ifstream f("C:/Users/Yuying/Desktop/train.csv");   
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
					date.push_back(field);
				}
				else
					fields.push_back(stringToNum(field));
				
			}
			Y.push_back(fields[fields.size() - 1]);      
        	Train.push_back(fields);
		} 
	}
}

void Init()
{
	cnt_layer = 2;
	cnt_times = 2;
	cnt_hidden_node = 10;
	Step = 0.00001;
	
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
	
	srand(time(NULL));
	for(int i = 0; i < X[0].size(); i++)
	{
		vector<double> w;
		for(int j = 0; j < cnt_hidden_node - 1; j++)
		{
			double p = rand()%(N+1)/(double)(N+1);
			w.push_back(p);
//			cout << p << " ";
		}
//		cout << endl;
		Wij.push_back(w);
	}	
//	cout << endl;
	
	for(int i = 0; i < cnt_hidden_node; i++)
	{
		double p = rand()%(N+1)/(double)(N+1);
		Wj.push_back(p);
//		cout << p << " ";
	}
//	cout << endl << endl;
}

//数据预处理
void DealWithData()
{
	for(int i = 0; i < X[0].size(); i++)
	{
		double min_num = 1000000;
		double max_num = X[0][i];
		for(int j = 0; j < X.size(); j++)
		{
			if(X[j][i] > max_num)
				max_num = X[j][i];
			else if(X[j][i] < min_num)
				min_num = X[j][i];
		}
		if(max_num > 1)
		{
			for(int j = 0; j < X.size(); j++)
			{
				X[j][i] = (X[j][i] - min_num) / (max_num - min_num);
			}
		}
	}
} 

vector<double> GetInH(int n)
{
	vector<double> InH;
	
	//h0
	InH.push_back(1);
	
	for(int i = 0; i < cnt_hidden_node - 1; i++)
	{
		double temp = 0;
		for(int k = 0; k < X[n].size(); k++)
		{
			temp += X[n][k] * Wij[k][i];
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
		OutH.push_back(Sigmoid(InH[i]));
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

vector<double> GetAllY()
{
	vector<double> preY;
	for(int i = 0; i < Y.size(); i++)
	{
		vector<double> InH = GetInH(i);
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
		double a = a0 * Wj[j] * DerivativeOfSig(OutH[j]);
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

//更新权重 
void GetNewW(int n)
{
	vector<vector<double>> CostOfWij;
	vector<double> CostOfWj;
	vector<double> InH = GetInH(n);
	vector<double> OutH = GetOutH(InH);
	double y = GetPreY(OutH);
	double a0 = GetA0(n, y);
//	cout << a0 << endl;
	vector<double> aj = GetAj(n, a0, OutH);
	for(int i = 0; i < Wij.size(); i++)
	{
		vector<double> cost;
		for(int j = 0; j < Wij[i].size(); j++)
		{
			cost.push_back(0);
		}
		CostOfWij.push_back(cost);
	}
	
//	cout << CostOfWij.size() << "    " << CostOfWij[0].size() << endl;
	
	for(int i = 0; i < Wj.size(); i++)
	{
		CostOfWj.push_back(0);
	}
	
	for(int i = 0; i < CostOfWj.size(); i++)
	{ 	
		CostOfWj[i] += a0 * OutH[i];
	//	cout << CostOfWj[i] << endl;
	}
//	cout << endl;
	
	for(int i = 0; i < CostOfWij.size(); i++)
 	{
 		for(int j = 0; j < CostOfWij[i].size(); j++)
 		{
 			CostOfWij[i][j] += aj[j] * X[n][i];
 		//	cout << aj[j] << endl;
 		}
 	//	cout << endl;
 	}
	
	for(int i = 0; i < CostOfWj.size(); i++)
	{
		Wj[i] += Step*CostOfWj[i];
	/*	cout << "Cost:" << endl;
		cout << CostOfWj[i] << endl;
		cout << "Wj:" << endl;
		cout << Wj[i] << endl;
		cout << endl;*/
	} 
//	cout << endl << endl;
	
	for(int i = 0; i < CostOfWij.size(); i++)
	{
		for(int j = 0; j < CostOfWij[i].size(); j++)
		{
			Wij[i][j] += Step*CostOfWij[i][j];
		}
	}
	
/*	cout << "a0" << endl << a0 << endl;
	cout << "aj:" << endl;
	for(int i = 0; i < aj.size(); i++)
	{
		cout << aj[i] << " ";
	}
	cout << endl;
	cout << endl;
	
	cout << "CostOfWj:" << endl;
	for(int i = 0; i < CostOfWj.size(); i++)
	{
		cout << CostOfWj[i] << " ";
	} 
	cout << endl << endl;
	
	cout << "CostOfWij:" << endl;
	for(int i = 0; i < CostOfWij.size(); i++)
	{
		for(int j = 0; j < CostOfWij[i].size(); j++)
			cout << CostOfWij[i][j] << " ";
		cout << endl;
	}
	cout << endl;
	
	cout << "Wj:" << endl;
	for(int i = 0; i < Wj.size(); i++)
		cout << Wj[i] << " ";
	cout << endl << endl;
	
	cout << "Wij:" << endl;
	for(int i = 0; i < Wij.size(); i++)
	{
		for(int j = 0; j < Wij[i].size(); j++)
			cout << Wij[i][j] << " ";
		cout << endl;
	}
	cout << endl;*/
	
	//训练集的预测结果 
/*	vector<double> pre_Y;
	for(int i = 0; i < OutH.size(); i++)
	{
		double y = 0;
		for(int j = 0; j < OutH[i].size(); j++)
		{
			y += Wj[j] * OutH[i][j];
		}
		pre_Y.push_back(y);
		
	}
	
	//计算误差梯度
	vector<double> A0;
	vector<vector<double>> Aj;
	for(int i = 0; i < Y.size(); i++)
	{
		double a0 = Y[i] - pre_Y[i];
		A0.push_back(a0); 
	} 
	
	for(int i = 0; i < Y.size(); i++)
	{
		vector<double> aj;
		for(int j = 1; j < OutH[i].size(); j++)
		{
			double a = A0[i] * Wj[j] * DerivativeOfSig(OutH[i][j]);
			aj.push_back(a);
		}
		Aj.push_back(aj);
	}
	
	//更新权重步长（取均值） 
	for(int i = 0; i < CostOfWj.size(); i++)
	{ 
		double avg0 = 0;
		for(int j = 0; j < Y.size(); j++)
		{
			avg0 += (double)A0[j] * OutH[j][i] / Y.size();
		}
		CostOfWj[i] += avg0;
	}
 	for(int i = 0; i < CostOfWij.size(); i++)
 	{
 		for(int j = 0; j < CostOfWij[i].size(); j++)
 		{
 			double avg = 0;
 			for(int k = 0; k < Y.size(); k++)
 			{
 				avg += Aj[k][j] * X[k][i] / Y.size();
 			}
 			CostOfWij[i][j] += avg;
 		}
 	}
 	
 	//更新权重
	for(int i = 0; i < CostOfWj.size(); i++)
	{
		Wj[i] += n*CostOfWj[i];
	} 
	
	for(int i = 0; i < CostOfWij.size(); i++)
	{
		for(int j = 0; j < CostOfWij[i].size(); j++)
		{
			Wij[i][j] += n*CostOfWij[i][j];
		}
	}*/ 
}

double MSE()
{
	double result = 0;
	vector<double> pre_Y = GetAllY();
	for(int i = 0; i < pre_Y.size(); i++)
	{
		result += (pre_Y[i] - Y[i]) * (pre_Y[i] - Y[i]);
		cout << pre_Y[i] << endl;
	}
	result = (double) result / Y.size();
	return result;
}

int main()
{
	Input();
	Init();
	DealWithData();
	while(cnt_times--)
	{
		for(int i = 0; i < Y.size(); i++) 
		{
			GetNewW(i);	
			
		}
		cout << MSE() << endl;
	}
	
	
/*	for(int i = 0; i < Wj.size(); i++)
		cout << Wj[i] << " ";
	cout << endl;
	
	for(int i = 0; i < Wij.size(); i++)
	{
		for(int j = 0; j < Wij[i].size(); j++)
			cout << Wij[i][j] << " ";
	}
	cout << endl;*/
	
}
