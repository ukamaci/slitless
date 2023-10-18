# 2023-01-18
# Ulas Kamaci

def outch_adjuster(out=None, true_out=None, outch_type=None, action=None):
    if action=='crop':
        if outch_type == 'int':
            y = true_out[:,[0]].clone()
        elif outch_type == 'vel':
            y = true_out[:,[1]].clone()
        elif outch_type == 'width':
            y = true_out[:,[2]].clone()
        elif outch_type == 'all':
            y = true_out
    elif action=='extend':
        y = true_out[:].clone()
        if outch_type == 'int':
            y[:,0] = out.detach().squeeze()
        elif outch_type == 'vel':
            y[:,1] = out.detach().squeeze()
        elif outch_type == 'width':
            y[:,2] = out.detach().squeeze()
        elif outch_type == 'all':
            y = out

    return y