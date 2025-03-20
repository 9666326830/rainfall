Source: [chrimerss/RainfallCamera](https://github.com/chrimerss/RainfallCamera/blob/f6c2029adff0785ed76dcc2e9360b173067840d3/data_process.py#L111-L195)

			self.labels= info.map_blocks(self.label, chunks=(1, ), drop_axis=[1,2], dtype=str ,name='label').compute()
			start_rain= time.time()
			array= array.map_blocks(self.crop, chunks=(1,400,300,3), dtype=np.uint8,name='crop')
			intensity= da.map_blocks(self.distribution, array, chunks=(1,), drop_axis=[1,2,3,4],
												 dtype=np.float32).compute()
			end_rain= time.time()
			print('processing rainfall costs ', round((end_rain-start_rain)/60.,2), ' minutes')
			# timeseries[]
			if first:
				df= pd.DataFrame(columns=['Rainfall'])
				first=False

			_df= pd.DataFrame(index=timerange, columns=['Rainfall'])
			_df.Rainfall= intensity
			df= pd.concat([df, _df])
			if OPT.logging_file:
				logging.info(_df)
				logging.info(f'processing one event containing {array.shape[0]} images \
				 				costs {round((end_rain-start_rain)/3600.,2)}  hours')
		end=time.time()
		print('total elapsed time :', round((end-start)/3600.), ' hours!')	
		return df

	def distribution(self, block, block_id=None):
		#input: dask array block
		#output: rainfall intensity
		img= block.squeeze().copy()
		label= self.labels[block_id[0]]
		if label== 'normal':
			intensity= self.normal(img)
		elif label=='night':
			intensity= self.night(img)
		elif label=='no rain':
			intensity= self.norain(img)
		elif label=='heavy':
			intensity= self.heavy(img)
		print(intensity)
		return np.array(intensity)[np.newaxis]

	def classfier_model(self):
		svm= load(self.class_model)

		return svm

	def rnn_model(self, model_path= './PReNet/logs/real/PReNet1.pth', recur_iter=4 ,use_GPU= OPT.use_GPU):
		model= Generator_lstm(recur_iter, use_GPU)
		if use_GPU:
			model = model.cuda()
		model.load_state_dict(torch.load(model_path, map_location='cpu'))
		model.eval()

		return model

	def label(self, block):
		info= block.squeeze().copy().reshape(1,-1)
		label= self.svm.predict(info)
		
		return np.array(label)[np.newaxis]

	def normal(self, src, use_GPU= OPT.use_GPU):
		# rainfall calculation under normal condition
		return RRCal().img_based_im(src, self.rnn, use_GPU=use_GPU)

	def heavy(self, src):
		# heavy rainfall regression model adds here
		return np.nan

	def norain(self, src):
		return 0

	def night(self, src):
		# night rainfall calculation adds here
		return np.nan

	def img_info(self, block):
		sub_img= block.squeeze().copy()
		m,n,c= sub_img.shape
		assert sub_img.shape[:2]==self.window_size, f'Input image has shape {sub_img.shape[:2]},\
														 but expect shape {self.window_size}'
		values= np.zeros(5, dtype=np.float32)
		#RMSE
		err=0
		for i in range(c):
			err+= (sub_img[:,:,i]-sub_img.mean(axis=2))**2
		values[0]= err.sum()/m/n/c