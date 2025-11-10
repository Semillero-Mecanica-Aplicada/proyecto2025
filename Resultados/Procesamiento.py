import numpy as np
from scipy.sparse import coo_matrix
import solidspy.assemutil as ass
import solidspy.solutil as sol
import solidspy.postprocesor as pos
from datetime import datetime

def create_grid(img, len_cell=1):
    len_mesh = np.shape(img)[0] + 1

    X, Y = np.meshgrid(np.linspace(0, len_cell, len_mesh), np.linspace(0, len_cell, len_mesh))
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    coords_raw = np.stack((X_flat, Y_flat), axis=1)
    nodal_id = np.array(range(len(coords_raw)))
    bc_flag_matrix = np.zeros((len(coords_raw), 2))
    coords_array = np.column_stack((nodal_id, coords_raw, bc_flag_matrix))
    return coords_array
def create_topology(img):
    len_mesh = np.shape(img)[0] + 1
    neles_side = np.shape(img)[0]
    ncoords_side = len_mesh

    eles_raw = np.empty((0,4),dtype=int)

    for j in np.arange(0, neles_side):
        for i in np.arange(0, neles_side):
            n0 = j * ncoords_side + i         
            n1 = n0 + 1                      
            n2 = n1 + ncoords_side           
            n3 = n0 + ncoords_side           
            eles_raw = np.vstack( (eles_raw, [n0, n1, n2, n3]) )
    img_assign = np.flip(img,axis=0).flatten()
    eles_id = np.array(range(len(eles_raw)))
    eles_type = np.ones(len(eles_raw))
    eles_array = np.column_stack((eles_id, eles_type, img_assign, eles_raw))
    return eles_array
def reassign_dof_periodicBC(ndofs, dofs_ima, dofs_ref):
    """Find dof number after applying boundary conditions

    Parameters
    ----------
    ndofs : int
        Number of degrees of freedom.
    dofs_ima : list, int
        List with image degrees of freedom.
    dofs_ref : list, int
        List with reference degrees of freedom.


    Examples
    --------
    
    Let us consider an example for  a square made with spring and masses.
    In that case we have the following dofs mappings.
    
    >>> dofs_ref = [0, 1, 0, 1, 0, 1]
    >>> dofs_ima = [2, 3, 4, 5, 6, 7]
    
    And the new numbering would be
    
    >>> new_num = reassign_dof_periodicBC(8, dofs_ima, dofs_ref)
    >>> new_num
    [0, 1, 0, 1, 0, 1, 0, 1]

    Returns
    -------
    new_num : list, int
        List with new numbering for the dof after applying Bloch analysis.

    """
    new_num = []
    cont = 0
    for cont_dof in range(ndofs):
        if cont_dof in dofs_ima:
            cont += 1
            ref_dof = dofs_ref[dofs_ima.index(cont_dof)]
            if ref_dof < cont_dof:
                new_num.append(ref_dof)
            else:
                new_num.append(ref_dof - cont)
        else:
            new_num.append(cont_dof - cont)
    return new_num


def periodicBC_transform_mat(def_const, ndofs, nodes_ref, nodes_ima,
                             dofs_ref, dofs_ima, nodes_ref_dof, nodes_ima_dof,
                             new_num,
                             nodes=None,
                             mode="tension_x",
                             disp_value=0.64):
    """
    Form transformation matrices for Periodic Boundary Conditions
    with imposed macro-deformations (tension_x, tension_y, shear).

    Parameters
    ----------
    def_const : float
        Default deformation constant for periodic relation.
    ndofs : int
        Total number of degrees of freedom.
    nodes_ref, nodes_ima : list[int]
        Reference and image nodes for periodic BC.
    dofs_ref, dofs_ima : list[int]
        DOFs corresponding to nodes_ref and nodes_ima.
    nodes_ref_dof, nodes_ima_dof : dict
        Dicts mapping node -> [ux_dof, uy_dof].
    new_num : list[int]
        New numbering after periodic renumbering.
    nodes : ndarray, optional
        Full node array (required to detect top/right faces).
        Assumed columns: [id, x, y].
    mode : str
        Type of macroscopic deformation.
        Options: 'tension_x', 'tension_y', 'shear'.
    disp_value : float
        Prescribed displacement magnitude.

    Returns
    -------
    Tmat : scipy.sparse.csr_matrix
        Transformation matrix.
    g_vec : ndarray
        Vector of prescribed displacements (periodic + imposed macro load).
    """

    rows, cols, vals = [], [], []
    g_vec = np.zeros(ndofs)

    # --- Inner nodes (no periodic relation) ---
    for cont_dof in range(ndofs):
        if cont_dof not in dofs_ima:
            rows.append(cont_dof)
            cols.append(new_num[cont_dof])
            vals.append(1)

    # --- Apply periodic BC relations + imposed macro displacement ---
    for node_ima, node_ref in zip(nodes_ima, nodes_ref):
        dof_ref = nodes_ref_dof[node_ref]
        dof_ima = nodes_ima_dof[node_ima]
        for cont_ref, cont_ima in zip(dof_ref, dof_ima):
            rows.append(cont_ima)
            cols.append(new_num[cont_ref])
            vals.append(1)
            g_vec[cont_ima] = def_const  # base periodic link

    # --- Apply macroscopic deformation ---
    if nodes is not None:
        x_max, x_min = np.max(nodes[:, 1]), np.min(nodes[:, 1])
        y_max, y_min = np.max(nodes[:, 2]), np.min(nodes[:, 2])

        for node in nodes_ima:
            dofs = nodes_ima_dof[node]
            ux, uy = dofs[0], dofs[1]
            x, y = nodes[node, 1], nodes[node, 2]

            # --- CASE 1: Tensión en X ---
            if mode == "tension_x" and np.isclose(x, x_max):
                g_vec[ux] += disp_value

            # --- CASE 2: Tensión en Y ---
            elif mode == "tension_y" and np.isclose(y, y_max):
                g_vec[uy] += disp_value

            # --- CASE 3: Corte puro (shear) ---
            elif mode == "shear":
                if np.isclose(y, y_max):   # cara superior
                    g_vec[ux] = disp_value
                elif np.isclose(y, y_min): # cara inferior
                    g_vec[ux] = 0.0

    nconds = len(dofs_ref)
    Tmat = coo_matrix((vals, (rows, cols)), shape=(ndofs, ndofs - nconds)).tocsr()
    return Tmat, g_vec

def periodicBC_nodes_identification(nodes):

    # Identificar id de los nodos referencia de la izquierda
    nodes_mask_id_L = (nodes[:,1] == np.min(nodes[:,1]))
    ref_nodes_L = nodes[:,0][nodes_mask_id_L]
    ref_nodes_L = np.delete(ref_nodes_L, -1)
    
    # Identificar id de los nodos imagen de la derecha
    nodes_mask_id_R = (nodes[:,1] == np.max(nodes[:,1]))
    ref_nodes_R = nodes[:,0][nodes_mask_id_R]
    ref_nodes_R= np.delete(ref_nodes_R, -1)
    
    # Identificar id de los nodos referencia de abajo
    nodes_mask_id_B = (nodes[:,2] == np.min(nodes[:,2]))
    ref_nodes_B = nodes[:,0][nodes_mask_id_B]
    ref_nodes_B = np.delete(ref_nodes_B, -1)
    
    # Identificar id de los nodos imagen de arriba
    nodes_mask_id_T = (nodes[:,2] == np.max(nodes[:,2]))
    ref_nodes_T = nodes[:,0][nodes_mask_id_T]
    ref_nodes_T = np.delete(ref_nodes_T, -1)
    
    # Identificar id del nodo referencia BL
    nodes_mask_id_BL = nodes[nodes_mask_id_L][:,2] == np.min(nodes[nodes_mask_id_L][:,2])
    ref_nodes_BL = nodes[:,0][nodes_mask_id_L][nodes_mask_id_BL]
    
    # Identificar id del nodo imagen TR
    nodes_mask_id_TR = nodes[nodes_mask_id_R][:,2] == np.max(nodes[nodes_mask_id_R][:,2])
    ref_nodes_TR = nodes[:,0][nodes_mask_id_R][nodes_mask_id_TR]

    nodes_ref =  np.int_(np.concatenate((ref_nodes_L, ref_nodes_B, ref_nodes_BL)))
    nodes_ima = np.int_(np.concatenate((ref_nodes_R, ref_nodes_T, ref_nodes_TR)))
    
    return nodes_ref, nodes_ima

def periodicBC_generate_inputs(nodes_id_list):
    node_dofs_dict = {}
    dofs_list = []
    for node in nodes_id_list:
        dofs = [2 * node, 2 * node + 1]
        
        dofs_list.extend(dofs)
        node_dofs_dict[node] = dofs
        
    return dofs_list, node_dofs_dict

mat1 = [831e6, 0.35] # Cancellous bone
mat2 = [12e9, 0.27] # Calcium Phosphates
mats = np.vstack((mat1,mat2))
imagenes = np.load("images0_125.npz")
images = imagenes["arr_0"]
nodes = create_grid(images[0], len_cell=1)
start_time = datetime.now()
results = np.zeros((400,6))
for i in range(1600,2000):
    img = images[i]
    elements = create_topology(img).astype(int)
    assem_op, bc_array, neq = ass.DME(nodes[:, -2:], elements)
    stiff_mat, _ = ass.assembler(elements, mats, nodes[:, :3], neq, assem_op)
    ndofs = 2*len(nodes)
    nodes_ref, nodes_ima = periodicBC_nodes_identification(nodes)
    dofs_ref, nodes_ref_dof = periodicBC_generate_inputs(nodes_ref)
    dofs_ima, nodes_ima_dof = periodicBC_generate_inputs(nodes_ima)
    new_num = reassign_dof_periodicBC(ndofs, dofs_ima, dofs_ref)
    disp_constant = 1
    t_mat_tx, g_vec_tx = periodicBC_transform_mat(disp_constant, ndofs, nodes_ref, nodes_ima,
                        dofs_ref, dofs_ima, nodes_ref_dof, nodes_ima_dof,
                        new_num,nodes=nodes,mode="tension_x")
    stiff_mat_hat_tx = t_mat_tx.T @ stiff_mat @ t_mat_tx
    rhs_hat_tx = t_mat_tx.T @ -stiff_mat @ g_vec_tx
    disp_hat_tx = sol.static_sol(stiff_mat_hat_tx.tocsr(), rhs_hat_tx)
    disp_tx = t_mat_tx @ disp_hat_tx + g_vec_tx
    disp_complete_tx = pos.complete_disp(bc_array, nodes, disp_tx)
    strain_nodes_tx, stress_nodes_tx = pos.strain_nodes(nodes, elements, mats,
                                            disp_complete_tx)
    t_mat_ty, g_vec_ty = periodicBC_transform_mat(disp_constant, ndofs, nodes_ref, nodes_ima,
                        dofs_ref, dofs_ima, nodes_ref_dof, nodes_ima_dof,
                        new_num,nodes=nodes,mode="tension_y")
    stiff_mat_hat_ty = t_mat_ty.T @ stiff_mat @ t_mat_ty
    rhs_hat_ty = t_mat_ty.T @ -stiff_mat @ g_vec_ty
    disp_hat_ty = sol.static_sol(stiff_mat_hat_ty.tocsr(), rhs_hat_ty)
    disp_ty = t_mat_ty @ disp_hat_ty + g_vec_ty
    disp_complete_ty = pos.complete_disp(bc_array, nodes, disp_ty)
    strain_nodes_ty, stress_nodes_ty = pos.strain_nodes(nodes, elements, mats,
                                            disp_complete_ty)
    t_mat_sh, g_vec_sh = periodicBC_transform_mat(disp_constant, ndofs, nodes_ref, nodes_ima,
                        dofs_ref, dofs_ima, nodes_ref_dof, nodes_ima_dof,
                        new_num,nodes=nodes,mode="shear")
    stiff_mat_hat_sh = t_mat_sh.T @ stiff_mat @ t_mat_sh
    rhs_hat_sh = t_mat_sh.T @ -stiff_mat @ g_vec_sh
    disp_hat_sh = sol.static_sol(stiff_mat_hat_sh.tocsr(), rhs_hat_sh)
    disp_sh = t_mat_sh @ disp_hat_sh + g_vec_sh
    disp_complete_sh = pos.complete_disp(bc_array, nodes, disp_sh)
    strain_nodes_sh, stress_nodes_sh = pos.strain_nodes(nodes, elements, mats,
                                            disp_complete_sh)
    strain = np.concatenate(
    [strain_nodes_tx.T, strain_nodes_ty.T, strain_nodes_sh.T],
    axis=1)
    stress = np.concatenate(
        [stress_nodes_tx.T, stress_nodes_ty.T, stress_nodes_sh.T],
        axis=1)
    strain[2,:] *= 0.5
    C = (stress @ strain.T) @ np.linalg.inv(strain @ strain.T)
    C = 0.5*(C + C.T)
    Vector = np.array([C[0,0], C[1,1], C[2,2], C[0,1], C[1,2], C[0,2]])
    results[i,:] = Vector
np.savez_compressed("Resultados_periodicBC.npz", results=results)