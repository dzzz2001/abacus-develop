#include "Problems/PoincareEmbeddings.h"

/*Define the namespace*/
namespace ROPTLIB {

	PoincareEmbeddings::PoincareEmbeddings(Vector indata, integer inn, integer inXNum, integer inNegSampleNum, realdp inSampleDampening)
	{
		data = indata;
		N = data.Getrow();			//the number of sample
		n = inn;						//embedding dimensions
		XNum = inXNum;					//the number of object
		NegSampleNum = inNegSampleNum;	//the number of negative samples
		SearchNum = 10 * NegSampleNum;	//the number of times negative samples were searched
		NegSampleNum++; //Note that this also includes the nodes in the positive sample
//		BatchSize = inbatchsize;
		SampleDampening = inSampleDampening;
//		BatchIndex = nullptr;
		NumGradHess = false;
//		burnin = true;
		
		dataToGraph();					//transform data into graphs
		createAliasTable();
		NegSamp = Vector(N * NegSampleNum);
		NegSamp = NegativeSampler();
		
	};

	PoincareEmbeddings::~PoincareEmbeddings(void)
	{
	};

	Vector &PoincareEmbeddings::NegativeSampler() const
	{
		realdp *NegSampptr = NegSamp.ObtainWriteEntireData();
		const realdp *dataptr = data.ObtainReadData();
		srand(std::random_device()());
		for (integer i = 0; i < N; i++)
		{
			integer u = static_cast<integer> (dataptr[i]);
			integer v = static_cast<integer> (dataptr[i + N]);
			NegSampptr[i * NegSampleNum] = v;
			integer SearchCnt = 0, NegSampCnt = 1;

			while (NegSampCnt < NegSampleNum && SearchCnt < SearchNum)
			{
				integer k = RandomNode();

				if (k != u && k != v
					&& dataGraph.at(u).find(k) == dataGraph.at(u).end()
					&& (dataGraph.find(k) == dataGraph.end() || dataGraph.at(k).find(u) == dataGraph.at(k).end()))
				{
					NegSampptr[i * NegSampleNum + NegSampCnt] = k;
					NegSampCnt++;
				}
				SearchCnt++;
			}

			/*If the number of sample sets is still insufficient after the above search, 
			the samples in the negative sample set are randomly repeated until the negative samples are filled up*/
			while (NegSampCnt < NegSampleNum)
			{
                integer k = static_cast<integer> (genrandreal() * NegSampCnt); //-- (integer)((double)rand() / ((double)RAND_MAX + 1) * NegSampCnt);
				NegSampptr[i * NegSampleNum + NegSampCnt] = NegSampptr[i * NegSampleNum + k];
				NegSampCnt++;
			}
		}
		NegSamp.Reshape(NegSampleNum, N).Transpose();

		return NegSamp;
	};

	realdp PoincareEmbeddings::f(const Variable &x) const
	{
		ProductManifold *ProdDomain = dynamic_cast<ProductManifold *> (Domain);
		realdp result = 0;
		const realdp *dataptr = data.ObtainReadData();
		Vector ExpDisti(NegSampleNum);
//        Vector ExpDist(1, &ExpDisti, BatchSize);
        Vector ExpDist(1, &ExpDisti, N);
		ExpDist.NewMemoryOnWrite();

//        Vector Z(BatchSize);
        Vector Z(N);
		realdp *Zptr = Z.ObtainWriteEntireData();

		Vector flag(XNum);
		flag.SetToZeros();
		realdp *flagptr = flag.ObtainWritePartialData();

		const realdp *NegSampptr = NegSamp.ObtainReadData();
//        x.Print("x:");//---
//        std::cout << "N:" << N << std::endl;//---
//        for (integer k = 0; k < BatchSize; k++)
        for (integer k = 0; k < N; k++)
		{
            integer t = k; //--- BatchIndex[k];
			integer i = static_cast<integer> (dataptr[t]);
			integer j = static_cast<integer> (dataptr[t + N]);
			realdp *ExpDistptr = ExpDist.GetElement(k).ObtainWriteEntireData();

			flagptr[i] = 1;
			flagptr[j] = 1;
            
//            std::cout << "i:" << i << ", j:" << j << std::endl;//---
			ExpDistptr[0] = std::exp(-ProdDomain->GetManifold(0)->Dist(x.GetElement(i), x.GetElement(j)));
			realdp Zi = ExpDistptr[0];

			//Note that j = NegSampleNum[t]
			for (integer u = 0; u < NegSampleNum - 1; u++)
			{
				integer negid = static_cast<integer> (NegSampptr[t + N * (u + 1)]);
				flagptr[negid] = 1;
				ExpDistptr[u + 1] = std::exp(-ProdDomain->GetManifold(0)->Dist(x.GetElement(i), x.GetElement(negid)));
				Zi += ExpDistptr[u + 1];
			}
			result -= std::log(ExpDistptr[0]);
			result += std::log(Zi);
			Zptr[k] = Zi;

		}
		x.AddToFields("ExpDist", ExpDist);
		x.AddToFields("Z", Z);
		x.AddToFields("flag", flag);
        result = result / N;
//        result = result / BatchSize;
		return result;
	};

    realdp PoincareEmbeddings::Stof(const Variable &x, const Vector &batch_index) const
    {
        ProductManifold *ProdDomain = dynamic_cast<ProductManifold *> (Domain);
        realdp result = 0;
        integer BatchSize = batch_index.Getlength();
        const realdp *dataptr = data.ObtainReadData();
        Vector ExpDisti(NegSampleNum);
        Vector ExpDist(1, &ExpDisti, BatchSize);
        ExpDist.NewMemoryOnWrite();

        Vector Z(BatchSize);
        realdp *Zptr = Z.ObtainWriteEntireData();

        Vector flag(XNum);
        flag.SetToZeros();
        realdp *flagptr = flag.ObtainWritePartialData();

        const realdp *NegSampptr = NegSamp.ObtainReadData();

        for (integer k = 0; k < BatchSize; k++)
        {
            integer t = static_cast<integer> (batch_index[k]);
            integer i = static_cast<integer> (dataptr[t]);
            integer j = static_cast<integer> (dataptr[t + N]);
            realdp *ExpDistptr = ExpDist.GetElement(k).ObtainWriteEntireData();

            flagptr[i] = 1;
            flagptr[j] = 1;

            ExpDistptr[0] = std::exp(-ProdDomain->GetManifold(0)->Dist(x.GetElement(i), x.GetElement(j)));
            realdp Zi = ExpDistptr[0];

            //Note that j = NegSampleNum[t]
            for (integer u = 0; u < NegSampleNum - 1; u++)
            {
                integer negid = static_cast<integer> (NegSampptr[t + N * (u + 1)]);
                flagptr[negid] = 1;
                ExpDistptr[u + 1] = std::exp(-ProdDomain->GetManifold(0)->Dist(x.GetElement(i), x.GetElement(negid)));
                Zi += ExpDistptr[u + 1];
            }
            result -= std::log(ExpDistptr[0]);
            result += std::log(Zi);
            Zptr[k] = Zi;

        }
        x.AddToFields("ExpDist", ExpDist);
        x.AddToFields("Z", Z);
        x.AddToFields("flag", flag);
        result = result / BatchSize;
        return result;
    };

	Vector &PoincareEmbeddings::EucGrad(const Variable &x, Vector *result) const
	{
		result->SetToZeros();
		const realdp *dataptr = data.ObtainReadData();
		Vector ExpDist = x.Field("ExpDist");
		Vector Z = x.Field("Z");
		const realdp *Zptr = Z.ObtainReadData();
		const realdp *NegSampptr = NegSamp.ObtainReadData();

//		for (integer k = 0; k < BatchSize; k++)
        for (integer k = 0; k < N; k++)
		{
            integer t = k; //--BatchIndex[k];
			integer i = static_cast<integer> (dataptr[t]);
			integer j = static_cast<integer> (dataptr[t + N]);
			
			const realdp *ExpDistiptr = ExpDist.GetElement(k).ObtainReadData();
			
			Vector left_grad(n), right_grad(n);

			// grad for positive example
			DistPartialDeriPoincare(x.GetElement(i), x.GetElement(j), &left_grad, &right_grad);
			result->GetElement(i).AlphaXaddThis((1 - ExpDistiptr[0] / Zptr[k])/N, left_grad);
			result->GetElement(j).AlphaXaddThis((1 - ExpDistiptr[0] / Zptr[k])/N, right_grad);

			// grad for negative example
			for (integer u = 0; u < NegSampleNum - 1; u++)
			{
				integer negid = static_cast<integer> (NegSampptr[t + N * (u + 1)]);
				DistPartialDeriPoincare(x.GetElement(i), x.GetElement(negid), &left_grad, &right_grad);
				result->GetElement(i).AlphaXaddThis(-ExpDistiptr[u + 1] / Zptr[k] / N, left_grad);
				result->GetElement(negid).AlphaXaddThis(-ExpDistiptr[u + 1] / Zptr[k] / N, right_grad);
			};
		};
		return *result;
	};

    Vector &PoincareEmbeddings::EucStoGrad(const Variable &x, Vector *result, const Vector &batch_index) const
    {
        result->SetToZeros();
        const realdp *dataptr = data.ObtainReadData();
        integer BatchSize = batch_index.Getlength();
        Vector ExpDist = x.Field("ExpDist");
        Vector Z = x.Field("Z");
        const realdp *Zptr = Z.ObtainReadData();
        const realdp *NegSampptr = NegSamp.ObtainReadData();

        for (integer k = 0; k < BatchSize; k++)
        {
            integer t = static_cast<integer> (batch_index[k]);
            integer i = static_cast<integer> (dataptr[t]);
            integer j = static_cast<integer> (dataptr[t + N]);
            
            const realdp *ExpDistiptr = ExpDist.GetElement(k).ObtainReadData();
            
            Vector left_grad(n), right_grad(n);

            // grad for positive example
            DistPartialDeriPoincare(x.GetElement(i), x.GetElement(j), &left_grad, &right_grad);
            result->GetElement(i).AlphaXaddThis((1 - ExpDistiptr[0] / Zptr[k])/BatchSize, left_grad);
            result->GetElement(j).AlphaXaddThis((1 - ExpDistiptr[0] / Zptr[k])/BatchSize, right_grad);

            // grad for negative example
            for (integer u = 0; u < NegSampleNum - 1; u++)
            {
                integer negid = static_cast<integer> (NegSampptr[t + N * (u + 1)]);
                DistPartialDeriPoincare(x.GetElement(i), x.GetElement(negid), &left_grad, &right_grad);
                result->GetElement(i).AlphaXaddThis(-ExpDistiptr[u + 1] / Zptr[k] / BatchSize, left_grad);
                result->GetElement(negid).AlphaXaddThis(-ExpDistiptr[u + 1] / Zptr[k] / BatchSize, right_grad);
            };
        };
        return *result;
    };

	void PoincareEmbeddings::DistPartialDeriPoincare(const Variable &x1, const Variable &x2, Vector *LEucGrad, Vector *REucGrad) const
	{
		LEucGrad->SetToZeros();
		REucGrad->SetToZeros();

		Vector u_minus_v(n);
		u_minus_v = x1 - x2;
		realdp u_minus_v2 = u_minus_v.DotProduct(u_minus_v);
		realdp uv = x1.DotProduct(x2);

		realdp uu = x1.DotProduct(x1);
		realdp vv = x2.DotProduct(x2);
		
		realdp alpha = 1.0 - uu;
		realdp beta = 1.0 - vv;
		realdp gamma = 1.0 + 2.0 * u_minus_v2 / alpha / beta;
        if (gamma <= 1) // for numerical error
            return;
//		if (gamma < 1)
//			gamma = 1; // for numerical error
//
//		if (gamma == 1)
//			return;

		realdp c = 4.0 / alpha / beta / std::sqrt(gamma * gamma - 1);

		// grad for the left parameter
		LEucGrad->AlphaXaddThis(c * (vv - 2.0 * uv + 1) / alpha, x1);
		LEucGrad->AlphaXaddThis(-c, x2);

		// grad for the right parameter
		REucGrad->AlphaXaddThis(c * (uu - 2.0 * uv + 1) / beta, x2);
		REucGrad->AlphaXaddThis(-c, x1);
	};

	void PoincareEmbeddings::dataToGraph()
	{
		const realdp *dataptr = data.ObtainReadData();
		for (int k = 0; k < N; k++)
		{
			integer i = static_cast<integer> (dataptr[k]);
			integer j = static_cast<integer> (dataptr[k + N]);

			if (dataGraph.find(i) != dataGraph.end())
				dataGraph[i].insert(j);
			else
				dataGraph.insert(std::pair<integer,std::unordered_set<integer>>(i, {j}));
		};
	};

	void PoincareEmbeddings::createAliasTable()
	{
		A = Vector(XNum);
		A.SetToZeros();
		S = Vector(XNum);

		realdp *APtr = A.ObtainWritePartialData();
		for (int i = 0; i < A.Getlength(); i++)
		{
			APtr[i] = i;
		}

		Vector counts(XNum);
		counts.SetToZeros();
		realdp *countsPtr = counts.ObtainWriteEntireData();
		const realdp *dataPtr = data.ObtainReadData();
		//Calculate the degree of the graph
		for (int i = 0; i < data.Getlength(); i++)
		{
			countsPtr[(integer)dataPtr[i]]++;
		}
		
		realdp countsSum = 0;
		for (int i = 0; i < counts.Getlength(); i++)
		{
			countsPtr[i] = pow(countsPtr[i], SampleDampening);
			countsSum += countsPtr[i];
		}

		S = counts.ScalarTimesThis(counts.Getlength() / countsSum);

		std::vector<integer> Tl, Th;
		const realdp *SReadPtr = S.ObtainReadData();
		for (integer i = 0; i < S.Getlength(); i++)
		{
			if (SReadPtr[i] < 1.0)
			{
				Tl.emplace_back(i);
			}
			else if (SReadPtr[i] > 1.0)
			{
				Th.emplace_back(i);
			}
		}

		realdp *SWritePtr = S.ObtainWritePartialData();
		
		while ((Tl.size() > 0) && (Th.size() > 0))
		{
			std::shuffle(Tl.begin(), Tl.end(), std::default_random_engine(std::random_device()()));
			std::shuffle(Th.begin(), Th.end(), std::default_random_engine(std::random_device()()));
			integer j = Tl.back();
			Tl.pop_back();
			integer k = Th.back();
			Th.pop_back();
			SWritePtr[k] = SReadPtr[k] - 1 + SReadPtr[j];
			APtr[j] = k;
			if (SReadPtr[k] < 1.0)
			{
				Tl.push_back(k);
			}
			else if (SReadPtr[k] > 1.0)
			{
				Th.push_back(k);
			}
		}
		
	};

	integer PoincareEmbeddings::RandomNode() const
	{
		if (burnin)
		{
			const realdp *SPtr = S.ObtainReadData();
			const realdp *APtr = A.ObtainReadData();
			realdp u = genrandreal() * XNum;
			integer fu = (integer)u;
			if (SPtr[fu] <= u - fu)
				return (integer)APtr[fu];
			else
				return fu;
		}
		else
		{
			return (integer)(genrandreal() * XNum);
		}
	};

	void PoincareEmbeddings::UpdateEpoch() const //integer *result
	{
//		Permutation(result);
		NegativeSampler();
	};


}; /*end of ROPTLIB namespace*/
