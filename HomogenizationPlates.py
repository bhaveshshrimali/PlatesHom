from dolfin import *
import numpy as np
import pandas as pd

set_log_level(10)
parameters["linear_algebra_backend"] = "PETSc"
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True
# parameters["krylov_solver"]["error_on_nonconvergence"] = False
# parameters["krylov_solver"]["maximum_iterations"] = 3000
# parameters["krylov_solver"]["monitor_convergence"] = True
ffc_options = {
    "optimize": True,
    "eliminate_zeros": True,
    "precompute_basis_const": True,
    "precompute_ip_const": True,
}
comm = MPI.comm_world

geom_type = "Hexagonal"
# geom_type = 'SimpleCubic'
fo = 0.75
L = 1.0
shp = "Ellipse"  # Square
alph = 1.2
if geom_type == "Hexagonal":
    Lx = np.sqrt(3.0) * L
    Ly = 2.0 * L
elif geom_type == "SimpleCubic":
    Lx = L
    Ly = L

vertices = np.array([[0.0, 0], [Lx, 0.0], [Lx, Ly], [0.0, Ly]])

area = Lx * Ly

# class used to define the periodic boundary map
class PeriodicBoundary(SubDomain):
    def __init__(self, vertices, tolerance=DOLFIN_EPS):
        """ vertices stores the coordinates of the 4 unit cell corners"""
        SubDomain.__init__(self, tolerance)
        self.tol = tolerance
        self.vv = vertices
        self.a1 = self.vv[1, :] - self.vv[0, :]  # first vector generating periodicity
        self.a2 = self.vv[3, :] - self.vv[0, :]  # second vector generating periodicity

    def inside(self, x, on_boundary):
        """return True if on left or bottom boundary AND NOT on one of the
        bottom-right or top-left vertices"""
        left = near(x[0], self.vv[0, 0])
        bottom = near(x[1], self.vv[0, 1])
        topleft = left and near(x[1], self.vv[3, 1])
        bottomright = bottom and near(x[0], self.vv[1, 0])

        return bool(left or bottom and not (topleft or bottomright))

    def map(self, x, y):
        """ Mapping the right boundary to left and top to bottom"""
        top = near(x[1], self.vv[3, 1])
        right = near(x[0], self.vv[1, 0])
        topright = top and right
        if topright:
            y[0] = x[0] - self.a1[0] - self.a2[0]
            y[1] = x[1] - self.a1[1] - self.a2[1]
        elif top and not (topright):
            y[0] = x[0] - self.a2[0]
            y[1] = x[1] - self.a2[1]
        elif right and not (topright):
            y[0] = x[0] - self.a1[0]
            y[1] = x[1] - self.a1[1]
        else:
            y[0] = -1.0
            y[1] = -1.0


class OriginPoint(SubDomain):
    def __init__(self, vertices, tolerance=DOLFIN_EPS):
        SubDomain.__init__(self, tolerance)
        self.vv = vertices

    def inside(self, x, on_boundary):
        return near(x[0], 0.0) and near(x[1], 0.0)


class bottomright(SubDomain):
    def __init__(self, vertices, tolerance=DOLFIN_EPS):
        SubDomain.__init__(self, tolerance)
        self.vv = vertices

    def inside(self, x, on_boundary):
        Lx = self.vv[:, 0].max()
        return near(x[0], Lx) and near(x[1], 0.0)


class topleft(SubDomain):
    def __init__(self, vertices, tolerance=DOLFIN_EPS):
        SubDomain.__init__(self, tolerance)
        self.vv = vertices

    def inside(self, x, on_boundary):
        Ly = self.vv[:, 1].max()
        return near(x[0], 0.0) and near(x[1], Ly)


def strain2voigt(eps):
    return as_vector([eps[0, 0], eps[1, 1], 2 * eps[0, 1]])


def voigt2stress(S):
    return as_tensor([[S[0], S[2]], [S[2], S[1]]])


def curv(theta):
    return sym(grad(theta))


def sstrn(w, theta):
    return theta - grad(w)


def bm(theta, nu, D, Gamm):
    DD = as_tensor([[D, nu * D, 0], [nu * D, D, 0], [0, 0, D * (1 - nu) / 2.0]])
    return voigt2stress(dot(DD, strain2voigt(curv(theta) + Gamm)))


def sf(w, theta, F):
    return F * sstrn(w, theta)


def macro_curv(i, scale):
    """returns the macroscopic curvature for the 3 elementary cases"""
    Gamm_Voigt = np.zeros((3,))
    Gamm_Voigt[i] = 1.0 * scale
    return np.array(
        [[Gamm_Voigt[0], Gamm_Voigt[2] / 2.0], [Gamm_Voigt[2] / 2.0, Gamm_Voigt[1]]]
    )


def stress2Voigt(s):
    return as_vector([s[0, 0], s[1, 1], s[0, 1]])


def build_nullspace(V, x):
    """Function to build null space for Plate problem
    Remember that V here is a mixed function space
    So V.sub(0) is w
    and V.sub(1) is theta
    """

    # Create list of vectors for null space
    # How many rigid body rotations do we need
    nullspace_basis = [x.copy() for i in range(3)]

    # Build translational null space basis
    V.sub(0).dofmap().set(nullspace_basis[0], 1.0)
    V.sub(1).sub(0).dofmap().set(nullspace_basis[1], 1.0)
    V.sub(1).sub(1).dofmap().set(nullspace_basis[2], 1.0)

    # Build rotational null space basis
    # V.sub(0).set_x(nullspace_basis[3], -1.0, 1)
    # V.sub(1).set_x(nullspace_basis[3],  1.0, 0)
    # V.sub(0).set_x(nullspace_basis[4],  1.0, 2)
    # V.sub(2).set_x(nullspace_basis[4], -1.0, 0)
    # V.sub(2).set_x(nullspace_basis[5],  1.0, 1)
    # V.sub(1).set_x(nullspace_basis[5], -1.0, 2)

    for x in nullspace_basis:
        x.apply("insert")

    # Create vector space basis and orthogonalize
    basis = VectorSpaceBasis(nullspace_basis)
    basis.orthonormalize()

    return basis


def generateCircleHex(rHex, L, Lx, Ly):
    lh1 = "pow(x[0],2)+pow(x[1]-" + str(0.5 * L) + ",2) < " + str(rHex ** 2)
    lh2 = "pow(x[0],2)+pow(x[1]-" + str(1.5 * L) + ",2) < " + str(rHex ** 2)
    bh1 = "pow(x[1],2)+pow(x[0]-" + str(0.5 * Lx) + ",2) < " + str(rHex ** 2)
    th1 = (
        "pow(x[1]-"
        + str(Ly)
        + ",2)+pow(x[0]-"
        + str(0.5 * Lx)
        + ",2) < "
        + str(rHex ** 2)
    )
    rh1 = (
        "pow(x[0]-"
        + str(Lx)
        + ",2)+pow(x[1]-"
        + str(0.5 * L)
        + ",2) < "
        + str(rHex ** 2)
    )
    rh2 = (
        "pow(x[0]-"
        + str(Lx)
        + ",2)+pow(x[1]-"
        + str(1.5 * L)
        + ",2) < "
        + str(rHex ** 2)
    )
    ch = (
        "pow(x[0]-"
        + str(Lx / 2.0)
        + ",2) + pow(x[1]-"
        + str(Ly / 2.0)
        + ",2) < "
        + str(rHex ** 2)
    )

    clist = [lh1, lh2, bh1, th1, rh1, rh2, ch]
    return clist


def generateEllipseHex(aHex, alph, L, Lx, Ly):
    bHex = alph * aHex
    aSq = "pow(" + str(aHex) + ",2)"
    bSq = "pow(" + str(bHex) + ",2)"
    lh1 = "pow(x[0],2)/" + aSq + "+pow(x[1]-" + str(0.5 * L) + ",2)/" + bSq + " < 1"
    lh2 = "pow(x[0],2)/" + aSq + "+pow(x[1]-" + str(1.5 * L) + ",2)/" + bSq + " < 1"
    bh1 = "pow(x[0]-" + str(0.5 * Lx) + ",2)/" + aSq + "+pow(x[1],2)/" + bSq + " < 0.99"
    th1 = (
        "pow(x[0]-"
        + str(0.5 * Lx)
        + ",2)/"
        + aSq
        + "+pow(x[1]-"
        + str(Ly)
        + ",2)/"
        + bSq
        + " < 0.99"
    )
    rh1 = (
        "pow(x[0]-"
        + str(Lx)
        + ",2)/"
        + aSq
        + "+pow(x[1]-"
        + str(0.5 * L)
        + ",2)/"
        + bSq
        + " < 1"
    )
    rh2 = (
        "pow(x[0]-"
        + str(Lx)
        + ",2)/"
        + aSq
        + "+pow(x[1]-"
        + str(1.5 * L)
        + ",2)/"
        + bSq
        + " < 1"
    )
    ch = (
        "pow(x[0]-"
        + str(0.5 * Lx)
        + ",2)/"
        + aSq
        + "+pow(x[1]-"
        + str(0.5 * Ly)
        + ",2)/"
        + bSq
        + " < 0.99"
    )

    clist = [lh1, lh2, bh1, th1, rh1, rh2, ch]
    return clist


def generateRectangleHex(s1Hex, alph, L, Lx, Ly):
    s2Hex = alph * s1Hex
    lh1 = (
        "abs(x[0]) < "
        + str(s1Hex / 2.0)
        + " && abs(x[1]-"
        + str(0.5 * L)
        + ") < "
        + str(s2Hex / 2.0)
    )
    lh2 = (
        "abs(x[0]) < "
        + str(s1Hex / 2.0)
        + " && abs(x[1]-"
        + str(1.5 * L)
        + ") < "
        + str(s2Hex / 2.0)
    )
    bh1 = (
        "abs(x[0]-"
        + str(0.5 * Lx)
        + ") < "
        + str(s1Hex / 2.0)
        + " && abs(x[1]) < "
        + str(s2Hex / 2.0)
    )
    th1 = (
        "abs(x[0]-"
        + str(0.5 * Lx)
        + ") < "
        + str(s1Hex / 2.0)
        + " && abs(x[1]-"
        + str(Ly)
        + ")"
        + " < "
        + str(s2Hex / 2.0)
    )
    rh1 = (
        "abs(x[0]-"
        + str(Lx)
        + ")"
        + " < "
        + str(s1Hex / 2.0)
        + " && abs(x[1]-"
        + str(0.5 * L)
        + ") < "
        + str(s2Hex / 2.0)
    )
    rh2 = (
        "abs(x[0]-"
        + str(Lx)
        + ")"
        + " < "
        + str(s1Hex / 2.0)
        + " && abs(x[1]-"
        + str(1.5 * L)
        + ") < "
        + str(s2Hex / 2.0)
    )
    ch = (
        "abs(x[0]-"
        + str(0.5 * Lx)
        + ")"
        + " < "
        + str(s1Hex / 2.0)
        + " && abs(x[1]-"
        + str(0.5 * Ly)
        + ") < "
        + str(s2Hex / 2.0)
    )

    clist = [lh1, lh2, bh1, th1, rh1, rh2, ch]
    return clist


def generateCircleCub(rCub, L, Lx, Ly):
    rSq = str(rCub ** 2)
    ch1 = "pow(x[0]-" + str(0.5 * Lx) + ",2)+pow(x[1]-" + str(0.5 * Ly) + ",2) < " + rSq

    return ch1


def generateEllipseCub(aCub, alph, L, Lx, Ly):
    bCub = alph * aCub
    aSq = "pow(" + str(aCub) + ",2)"
    bSq = "pow(" + str(bCub) + ",2)"
    ch1 = (
        "pow(x[0]-"
        + str(0.5 * Lx)
        + ",2)/"
        + aSq
        + "+pow(x[1]-"
        + str(0.5 * Ly)
        + ",2)/"
        + bSq
        + " < 1"
    )

    return ch1


def generateRectangle(s1Cub, alph, L, Lx, Ly):
    s2Cub = alph * s1Cub
    ch1 = (
        "abs(x[0]-"
        + str(0.5 * Lx)
        + ") < "
        + str(0.5 * s1Cub)
        + " && abs(x[1] - "
        + str(0.5 * Ly)
        + ") < "
        + str(0.5 * s2Cub)
    )

    return ch1


def getWithoutHoles(E, nu):
    # E = 2.2e9
    # nu=0.46
    mu = E / 2.0 / (1.0 + nu)
    lam = 2 * mu * nu / (1 - 2.0 * nu)
    iden = np.eye(3)

    # Lijkl = mu*(np.einsum('ik,jl->ijkl',iden2,iden2) + np.einsum('il,jk->ijkl',iden2,iden2)) + \
    #         lam*np.einsum('ij,kl->ijkl',iden2,iden2)
    iden2 = np.eye(2)
    L_abyd = mu * (
        np.einsum("ik,jl->ijkl", iden2, iden2) + np.einsum("il,jk->ijkl", iden2, iden2)
    ) + lam * np.einsum("ij,kl->ijkl", iden2, iden2)
    L_1212 = 2.0 * mu
    L_ab33 = lam * iden2
    fac = np.einsum("ij,kl->ijkl", L_ab33, L_ab33) / (lam + 2.0 * mu)
    Mtil = L_abyd - fac

    return Mtil, L_1212


"""
`Mesh` and Material parameters:
`Emat` and `nu_mat` for the matrix
`Eh` and `nu_h` for the inclusion """

bot_right = bottomright(vertices)
orgn = OriginPoint(vertices)
top_lft = topleft(vertices)


p0 = Point(0.0, 0.0)
p1 = Point(Lx, Ly)
msh = RectangleMesh.create(
    comm, [p0, p1], [400, 400], CellType.Type.quadrilateral
)  # add MPI communicator
# msh=UnitSquareMesh.create(100,100,CellType.Type.quadrilateral)
deg = 2
factr = 1.0e-7
nu_mat = 0.46
mu_m = 1.0e9
Emat = 2.2e9  # 2.*mu_m*(1+nu_mat)
mu_h = factr * mu_m
nu_h = factr * nu_mat
Eh = factr * Emat  # 2.*mu_h*(1+nu_h)
rHex = L * np.sqrt(
    fo * np.sqrt(3.0) / 2.0 / np.pi
)  # radius of each circle for a hexagonal distribution
aHex = L * np.sqrt(np.sqrt(3.0) * fo / 2.0 / np.pi / alph)
s1Hex = L * np.sqrt(np.sqrt(3.0) * fo / 2.0 / alph)
rCub = L * np.sqrt(fo / np.pi)
aCub = L * np.sqrt(fo / np.pi / alph)
s1Cub = L * np.sqrt(fo / alph)
# print(s1Cub)
# For Hexagonal distribution of pores
if shp == "Circle":
    Estring = (
        " || ".join(i for i in generateCircleHex(rHex, L, Lx, Ly))
        + " ? "
        + str(Eh)
        + " : "
        + str(Emat)
    )
    nustring = (
        " || ".join(i for i in generateCircleHex(rHex, L, Lx, Ly))
        + " ? "
        + str(nu_h)
        + " : "
        + str(nu_mat)
    )
elif shp == "Ellipse":
    Estring = (
        " || ".join(i for i in generateEllipseHex(aHex, alph, L, Lx, Ly))
        + " ? "
        + str(Eh)
        + " : "
        + str(Emat)
    )
    nustring = (
        " || ".join(i for i in generateEllipseHex(aHex, alph, L, Lx, Ly))
        + " ? "
        + str(nu_h)
        + " : "
        + str(nu_mat)
    )
elif shp == "Square" or shp == "Rectangle":
    Estring = (
        " || ".join(i for i in generateRectangleHex(s1Hex, alph, L, Lx, Ly))
        + " ? "
        + str(Eh)
        + " : "
        + str(Emat)
    )
    nustring = (
        " || ".join(i for i in generateRectangleHex(s1Hex, alph, L, Lx, Ly))
        + " ? "
        + str(nu_h)
        + " : "
        + str(nu_mat)
    )

# For a Simple Cubic distribution of pores
# if shp == 'Circle':
#     Estring = generateCircleCub(rCub,L,Lx,Ly) + ' ? ' + str(Eh) +' : '+str(Emat)
#     nustring = generateCircleCub(rCub,L,Lx,Ly) + ' ? ' + str(nu_h) +' : '+str(nu_mat)
# elif shp == 'Ellipse':
#     Estring = generateEllipseCub(aCub,alph,L,Lx,Ly) + ' ? ' + str(Eh) +' : '+str(Emat)
#     nustring = generateEllipseCub(aCub,alph,L,Lx,Ly) + ' ? ' + str(nu_h) +' : '+str(nu_mat)
# elif shp == 'Square' or shp == 'Rectangle':
#     Estring = generateRectangle(s1Cub,alph,L,Lx,Ly) + ' ? ' + str(Eh) +' : '+str(Emat)
#     nustring = generateRectangle(s1Cub,alph,L,Lx,Ly) + ' ? ' + str(nu_h) +' : '+str(nu_mat)

Evals = Expression(Estring, degree=2 * deg)
nuvals = Expression(nustring, degree=2 * deg)

# Evals = Expression(str(Emat),degree=0)
# nuvals = Expression(str(nu_mat),degree=0)

scl = 1.0e-2
thick = Constant(5.0e-3)
Dvals = Evals * thick ** 3 / (1 - nuvals ** 2) / 12.0
Fvals = Evals / 2.0 / (1 + nuvals) * thick * 5.0 / 6.0
Gamm_bar = Constant(((0.0, 0.0), (0.0, 0.0)))

dxs = dx(metadata={"quadrature_degree": 2 * deg - 2})
We = FiniteElement("CG", msh.ufl_cell(), deg - 1)
Te = VectorElement("CG", msh.ufl_cell(), deg)

# Re_s = FiniteElement('R',msh.ufl_cell(),0)
# Re_v = VectorElement('R',msh.ufl_cell(),0)

# Me = MixedElement([We,Te,Re_s,Re_v])
Me = MixedElement([We, Te])
Ve = FunctionSpace(msh, Me, constrained_domain=PeriodicBoundary(vertices))

# w, thta, lamb_w, lamb_thta = TrialFunctions(Ve)
# w_, thta_, lamb_w_, lamb_thta_ = TestFunctions(Ve)

"""
    Preventing Rigid body motions by setting the fields
    to zero at some points, alternative to using 
    lagrange multiplier to set zero average.

"""

w, thta = TrialFunctions(Ve)
w_, thta_ = TestFunctions(Ve)

w_thta = Function(Ve)
null_space = build_nullspace(Ve, w_thta.vector())

bc1 = DirichletBC(Ve.sub(0), Constant(0.0), orgn, method="pointwise")
bc2 = DirichletBC(Ve.sub(0), Constant(0.0), bot_right, method="pointwise")
bc3 = DirichletBC(Ve.sub(0), Constant(0.0), top_lft, method="pointwise")
bc4 = DirichletBC(Ve.sub(1), Constant((0.0, 0.0)), orgn, method="pointwise")

# bc1 = DirichletBC(Ve.sub(0),Constant(0.),orgn,method='pointwise')
# bc2 = DirichletBC(Ve.sub(1),Constant((0.,0.)),orgn,method='pointwise')
# bcs = [bc1,bc2]


bcs = [bc1, bc2, bc3, bc4]
# bcs = []
a_mu_v = (
    inner(bm(thta, nuvals, Dvals, Gamm_bar), curv(thta_)) * dx
    + inner(sf(w, thta, Fvals), sstrn(w_, thta_)) * dxs
)
# a_mu_v += inner(lamb_w,w_)*dx + inner(lamb_w_,w)*dx
# a_mu_v += inner(lamb_thta,thta_)*dx + inner(lamb_thta_,thta)*dx

L_w_thta, f_thta = lhs(a_mu_v), rhs(a_mu_v)

# Try improving convergence of the Krylov Solver
A, b = assemble_system(L_w_thta, f_thta, bcs, form_compiler_parameters=ffc_options)
# as_backend_type(A).set_near_nullspace(null_space)

# pc = PETScPreconditioner("petsc_amg")
# PETScOptions.set("mg_levels_ksp_type", "chebyshev")
# PETScOptions.set("mg_levels_pc_type", "jacobi")
# solver = PETScKrylovSolver("gmres", pc)
solver = PETScLUSolver(comm, "mumps")
# PETScOptions.set("mg_levels_esteig_ksp_type", "gmres")
# PETScOptions.set("mg_levels_ksp_chebyshev_esteig_steps", 100)
solver.set_operator(A)

# Coordinates to visualize complete displacement profile
x = SpatialCoordinate(msh)
x0 = as_vector([0.5 * Lx, 0.5 * Ly])
y_scl = x  # - x0
B_hom = np.zeros((3, 3))

for (j, case) in enumerate(["Kxx", "Kyy", "Kxy"]):

    print("Solving {} case...".format(case))
    Gamm_bar.assign(Constant(macro_curv(j, scl)))
    solver.solve(w_thta.vector(), b)
    w, thta = w_thta.split(True)

    # print('kapp_xx = {}'.format( float(assemble((curv(thta)+Gamm_bar)[0,0]*dx))/area ))
    # print('kapp_xy = {}'.format( float(assemble((curv(thta)+Gamm_bar)[0,1]*dx))/area ))
    # print('kapp_yy = {}'.format( float(assemble((curv(thta)+Gamm_bar)[1,1]*dx))/area ))

    M_til = np.zeros((3,))
    Kapp_til = M_til.copy()
    for k in range(3):
        M_til[k] = (
            float(assemble(stress2Voigt(bm(thta, nuvals, Dvals, Gamm_bar))[k] * dx))
            / area
        )
        Kapp_til[k] = (
            float(assemble(strain2voigt(curv(thta) + Gamm_bar)[k] * dx)) / area
        )

    B_hom[j, :] = M_til.copy() / scl
    # pd.DataFrame(M_til).to_excel(
    #     "/home/bshrima2/PlatesTrial/Mtil_{}_dBCf_mod.xlsx".format(case),
    #     index=False,
    #     header=False,
    # )
    # pd.DataFrame(Kapp_til).to_excel(
    #     "/home/bshrima2/PlatesTrial/kapp_{}_dBCf_mod.xlsx".format(case),
    #     index=False,
    #     header=False,
    # )

    w_full = w + 0.5 * dot(y_scl, dot(Gamm_bar, y_scl))
    thta_full = thta + dot(Gamm_bar, y_scl)

    Vw = FunctionSpace(msh, FiniteElement("CG", msh.ufl_cell(), deg))
    Vt = FunctionSpace(msh, VectorElement("CG", msh.ufl_cell(), deg))
    w_plot = Function(Vw, name="disp")
    thta_plot = Function(Vt, name="thta")
    Eplot = Function(Vw, name="Emat")

    w_plot.assign(project(w_full, Vw))
    thta_plot.assign(project(thta_full, Vt))
    Eplot.interpolate(Evals)

    with XDMFFile(
        comm,
        "/home/bshrima2/PlatesTrial/{}_Hex_case_dBCf_mod_{}.xdmf".format(
            geom_type, case
        ),
    ) as res_fil:
        res_fil.parameters["flush_output"] = True
        res_fil.parameters["functions_share_mesh"] = True
        res_fil.write(w_plot, 0)
        res_fil.write(thta_plot, 0)
        res_fil.write(Eplot, 0)

Btilde = B_hom * 3.0 / 2.0 / float(thick / 2.0) ** 3
M_solid, L1212_solid = getWithoutHoles(Emat, nu_mat)
Btilde[0, 0] /= M_solid[0, 0, 0, 0]
Btilde[0, 1] /= M_solid[0, 0, 1, 1]
Btilde[1, 0] /= M_solid[1, 1, 0, 0]
Btilde[1, 1] /= M_solid[1, 1, 1, 1]
Btilde[2, 2] /= M_solid[0, 1, 0, 1]
# print(np.array_str(B_hom*3./2/float(thick)**3, precision=3))
# np.savetxt('/home/bshrima2/PlatesTrial/kapp_xy')
# pd.DataFrame(Btilde).to_excel(
#     "/home/bshrima2/PlatesTrial/Btilde_{}_Hex_dBCf_mod.xlsx".format(geom_type),
#     index=False,
#     header=False,
# )
