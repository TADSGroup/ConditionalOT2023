    plt.subplot(1,3,1)
    plt.title(r"$\nu(y,u)$")
    plt.ylabel("u")
    plt.xlabel("y")
    plt.scatter(true_target[:,0],true_target[:,1],s=0.75,alpha=0.25)
    for y,color,ls in zip(yval_cond_points,colors,line_styles):
        plt.plot([y,y],[-1,1],c=color,ls=ls)

    plt.subplot(1,3,2)
    plt.title("$T(y,\cdot) \sharp \eta(v\mid y)$")
    plt.xlabel("v")
    for y,color,ls in zip(yval_cond_points,colors,line_styles):
        cond_samples = cot_model.get_conditional_samples(np.array([y]),50000)[:,0]
        kdeplot(cond_samples,bw_adjust=0.9,c=color,label = f'y = {y}, generated')
        eps = 0.01
        true_cond = sample_true_cond(y,dataset_name,eps,preprocessor,50000,batch_size = 500000)
        
        kdeplot(true_cond[:,1],bw_adjust=0.9,c=color,label = f'y = {y}, true',ls = ':')
        print(len(true_cond))
        plt.legend()

    plt.subplot(1,3,3)
    plt.title("$T(y,v)$")
    plt.xlabel("v")
    for y,color,ls in zip(yval_cond_points,colors,line_styles):
        grid = np.linspace(0,1,1000)
        fval = cot_model.eval_at_cond(np.array([y]),grid)[:,0]
        plt.plot(grid,fval,c=color,label = f'y = {y}')
        plt.legend()
