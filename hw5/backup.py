def coordinateDescent(Q,b,x,lamb,thr):
    p = Q.shape[0]
    iis = np.arange(p)
    tmp = np.copy(x)
    loss = [lassoLoss(Q,b,tmp,lamb)]
    while True:
        for j in iis:
            step_iis = np.delete(iis,j)
            tmp[j] = softThreshold(deltaX(Q[step_iis,j],b[j],tmp[step_iis]), lamb)/Q[j,j]
            loss.append(lassoLoss(Q,b,tmp,lamb))
        if np.abs(loss[-1] - loss[-2]) <= thr:
            break
    return tmp, loss
