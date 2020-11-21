#pragma once

/// <summary>
/// Spline c�bica param�trica natural, onde S''(t=0) = 0; S''(t=1) = 0
/// onde t(i) = i/N, portanto 0 < t < N
/// </summary>
class spline {
private:

	uint64_t N;								//N�mero de equa��es
	Eigen::VectorXd x, y;					// Coordenadas x e y dos pontos de entrada
	Eigen::MatrixXd A;						// AX=B
	Eigen::VectorXd X_x, X_y, B_x, B_y;		// AX=B, temos de fazer os c�lculos para x e para y
	Eigen::VectorXd h;						// h � o vetor que cont�m "t(i) - t(i-1)"
	Eigen::MatrixXd xCoeff, yCoeff;			// Coeficientes dos polin�mios
	Mat cvCoeff;
public:
	spline();
	spline(Eigen::VectorXd x, Eigen::VectorXd y);
	/// <summary>
	/// Armazena as coordenadas dos pontos
	/// </summary>
	/// <param name="x">coordenada x</param>
	/// <param name="y">coordenada y</param>
	void setPointCoord(Eigen::VectorXd x_, Eigen::VectorXd y_);
	/// <summary>
	/// h � o vetor que cont�m "t(i) - t(i-1)"
	/// considerando t(i) = i/N, portanto 0 < t < N
	/// o intervalo entre t(i) e t(i-i) � igualmente espa�ado
	/// e podemos dizer que o espa�amento � de t=1/N
	/// </summary>
	void setDifferenceH();
	/// <summary>
	/// Prepara a matriz A, que ser� utilizada na resolu��o do sistema: AX = B.
	/// A matriz A s� depende de t, por isso � a mesma matriz para as splines em x e em y.
	/// por isso criei apenas uma matriz A.
	/// </summary>
	void setMatrixA();
	/// <summary>
	/// Prepara o vetor B, que ser� utilizado na resolu��o do sistema AX = B.
	/// O vetor B depende de x e y, por isso n�o � o mesmo para as splines em x e y.
	/// por isso criei B_x e B_y.
	/// </summary>
	void setVectorB();
	/// <summary>
	/// Calcula o sistema linear AX = B
	/// </summary>
	void solveEquations();
	/// <summary>
	/// Armazena os valores dos coeficientes a, b, c, d tanto para as splines em x, quanto para as splines em y
	/// </summary>
	void setCoefficients();
	/// <summary>
	/// Calcula o resultado de uma fun��o param�trica c�bica.
	/// Onde a fun��o varia em t e tem retornos em x e em y.
	/// </summary>
	/// <param name="t">vati�vel de entrada </param>
	/// <param name="tk">constante</param>
	/// <param name="line">linha onde se encontram os coeficientes: a, b, c, d</param>
	/// <param name="x">resultado do polin�mio em x</param>
	/// <param name="y">resultado do polin�mio em y</param>
	void cubicFunction(double t, double tk, int line, double& x, double& y);
	/// <summary>
	/// Desenha as splines em uma imagem do OpenCV
	/// </summary>
	/// <param name="img">imagem onde ser�o desenhadas as splines </param>
	void drawSplines(Mat& img);

	Mat getCoefficients();
};

