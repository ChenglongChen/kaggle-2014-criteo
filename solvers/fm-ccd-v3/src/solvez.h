
inline float solve_z(
    float const * const Y,
    float const * const S,
    float const * const A,
    float const z_init,
    float const lambda,
    size_t const nr_instance)
{
    if(nr_instance == 0)
        return z_init;
	double z = z_init;
	double z_new = 0;
	double f = 0;
	double f_new = 0;
	double g = 0;
	double h = 0;
	double d = 0;
	double exp_dec = 0;
	const double beta = 0.5;
	const double gamma = 0.1f;
	const size_t max_iter = 2;

	for(size_t t = 1; t <= max_iter; t++){
		f = lambda / 2 * z * z;
		g = lambda*z;
		h = lambda;
		for(size_t i = 0; i <= nr_instance - 1; i++){
			exp_dec = std::exp(-Y[i] * (S[i] + z * A[i]));
			f += std::log(1 + exp_dec);
			g += -Y[i] * A[i] * exp_dec / (1 + exp_dec);
			h += exp_dec * pow(A[i] / (1 + exp_dec), 2);
		}
		d = -g / h;
		
        size_t nr_line_search = 0;
		do{
			z_new = z + d;
			f_new = lambda / 2 * z_new * z_new;
			for(size_t i = 0; i <= nr_instance - 1; i++)
				f_new += std::log( 1 + std::exp(-Y[i] * (S[i] + z_new * A[i])));
			d *= beta;
            ++nr_line_search;
            if(nr_line_search > 10)
                break;
		}while(f_new - f > gamma * d  * g);
	}
	return static_cast<float>(z_new);
}
