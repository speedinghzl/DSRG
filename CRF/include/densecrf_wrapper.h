#include "densecrf.h"

class DenseCRFWrapper {
	public:
		DenseCRFWrapper(int W, int H, int nlabels);
		virtual ~DenseCRFWrapper();

		void set_unary_energy(float* unary_costs_ptr);

		void add_pairwise_energy(float w1,
				         float theta_alpha_1, float theta_alpha_2,
				         float theta_betta_1, float theta_betta_2, float theta_betta_3,
					 float w2,
				         float theta_gamma_1, float theta_gamma_2,
				         unsigned char* im);

		void map(int n_iters, int* result);
		void inference(int n_iters, float* result);

		int npixels();
		int nlabels();

	private:
		DenseCRF2D* m_crf;
		int H;
		int W;
		int m_nlabels;
};
