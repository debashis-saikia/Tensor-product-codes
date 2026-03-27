F := GF(2);

Read("/mnt/c/Users/IISER13/OneDrive/Desktop/H_X.g");
Read("/mnt/c/Users/IISER13/OneDrive/H_Z.g");


H_X := List(H_X, row -> List(row, x -> Int(x) mod 2));
H_Z := List(H_Z, row -> List(row, x -> Int(x) mod 2));

P := Hx * TransposedMat(Hz);

Print("H_X * H_Z^T = \n");
Display(P);

if P = NullMat(8,8,F) then
    Print("Orthogonal \n");
else
    Print("NOT orthogonal \n");
fi;

rX := RankMat(H_X);
rZ := RankMat(H_Z);

n := Length(H_X[1]);
k := n - rX - rZ;

Print("n = ", n, "\n");
Print("k = ", k, "\n");

DistRandCSS(H_Z, H_X, 200, 0, 4 : field:=F);
