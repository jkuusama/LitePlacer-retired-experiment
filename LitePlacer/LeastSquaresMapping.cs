using System;
using System.Collections.Generic;
using System.Linq;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;

namespace LitePlacer {
    // back computes a rigid transformation 
    // based on http://nghiaho.com/?page_id=671
    // and https://en.wikipedia.org/wiki/Kabsch_algorithm
    public class LeastSquaresMapping {
        readonly List<PartLocation> source;
        readonly List<PartLocation> dest;
        private Matrix<double> Rotation, Offset;

        public LeastSquaresMapping(List<PartLocation> from, List<PartLocation> to) { 
            source = from;
            dest = to;
            //Recompute();
            Recompute2();
        }

        public override string ToString() {
            return String.Format("Offset = {0}\nAngle = {1}",
                new PartLocation(Offset) - Global.Instance.Locations.GetLocation("PCB Zero"),
                Angle);
        }

        public double Angle { get { return  Math.Asin(Rotation[0, 1]) * -180d / Math.PI; } }
        private Matrix<double> _sc, _dc;
        private Matrix<double> SourceCentroid { get { if (_sc == null) _sc = PartLocation.Average(source).ToMatrix(); return _sc; } }
        private Matrix<double> DestCentroid { get { if (_dc == null) _dc = PartLocation.Average(dest).ToMatrix(); return _dc; } }

        public Matrix<double> P{
            get {
                Matrix<double> p = new Matrix<double>(source.Count, 2);
                for (int i=0; i<source.Count; i++) { p[i, 0] = source[i].X;  p[i, 1] = source[i].Y;}
                return p;
            }
        } 

        public Matrix<double> Q{
            get {
                Matrix<double> p = new Matrix<double>(dest.Count, 2);
                for (int i=0; i<dest.Count; i++) { p[i, 0] = dest[i].X; p[i, 1] = dest[i].Y;  }
                return p;
            }
        }

        public string FormatMatrix(Matrix<double> m) {
            string s = "";
            for (int i = 0; i < m.Rows; i++) {
                for (int j = 0; j < m.Cols; j++) {
                    s += String.Format("{0:F1} \t", m[i, j]);
                }
                s += "\r\n";
            }
            return s;
        }


        // this is directly from the wikipedia page on Kabsh Algorithm
        public void Recompute2() {
            var p = P;
            var q = Q;

            //1. subtract centroids
            for (int i = 0; i < p.Rows; i++) {
                p[i, 0] -= SourceCentroid[0, 0];
                p[i, 1] -= SourceCentroid[1, 0];
                q[i, 0] -= DestCentroid[0, 0];
                q[i, 1] -= DestCentroid[1, 0];
            }

            //2. compute covariance matrix
            var a = p.Transpose()*q;

            //3. compute rotation matrix
            /* perform svd  where A =  V S WT */
            Matrix<double> V = new Matrix<double>(2, 2);
            Matrix<double> S = new Matrix<double>(2, 2);
            Matrix<double> W = new Matrix<double>(2, 2);
            CvInvoke.cvSVD(a.Ptr, S.Ptr, V.Ptr, W.Ptr, SVD_TYPE.CV_SVD_DEFAULT);

            // Deal with reflection matrix
            Matrix<double> m = new Matrix<double>(2, 2);
            m.SetIdentity(new MCvScalar(1));
            m[1,1] = ((W*V.Transpose()).Det<0) ? -1 : 1;

            // Comput the rotation matrix
            Rotation = W*m*V.Transpose();
            //Offset = DestCentroid - (Rotation * SourceCentroid);
            Offset = DestCentroid - SourceCentroid;

            Console.WriteLine("Rotaiton Matrix - Angle ="+Angle);
            Console.WriteLine(FormatMatrix(Rotation));
        }


        // this is from the blog entry
        public void Recompute() {
            if (source == null || dest == null || source.Count != dest.Count)
                throw new Exception("Input data null or not equal in length");

            // compute covariance matrix
            Matrix<double> H = new Matrix<double>(2, 2);
            
            H.SetZero();
            for (int i = 0; i < source.Count; i++) {
                var a = source[i].ToMatrix() - SourceCentroid;
                var b = dest[i].ToMatrix() - DestCentroid;
                H += a * b.Transpose();
            }
            
            /* perform svd  where A =  U W VT
             *  A  IntPtr  Source MxN matrix
             *  W  IntPtr  Resulting singular value matrix (MxN or NxN) or vector (Nx1).
             *  U  IntPtr  Optional left orthogonal matrix (MxM or MxN). If CV_SVD_U_T is specified, the number of rows and columns in the sentence above should be swapped
             *  V  IntPtr  Optional right orthogonal matrix (NxN)
             */
        
            Matrix<double> U = new Matrix<double>(2, 2); 
            Matrix<double> W = new Matrix<double>(2, 2);            
            Matrix<double> V = new Matrix<double>(2, 2);
            CvInvoke.cvSVD(H.Ptr, W.Ptr, U.Ptr, V.Ptr, SVD_TYPE.CV_SVD_DEFAULT);

            // compute rotational matrix R=V*UT
            Rotation = V * U.Transpose();            

            // find translation
            //Offset = DestCentroid - ( Rotation * SourceCentroid);
            Offset = DestCentroid - SourceCentroid;

            if (Angle > 5) {
                Global.Instance.mainForm.ShowSimpleMessageBox("Excessive Angle Detected - Problem detecting rotation\nOffset = " + new PartLocation(Offset-DestCentroid) + "\nAngle=" + Angle);
            }
        }

        /// <summary>
        /// Map a source point to a destination point based on the calibrated inputs
        /// </summary>
        public PartLocation Map(PartLocation from) {
            if (Rotation == null || Offset == null) throw new Exception("LeastSquareMapping not intialized");

            var x = from.ToMatrix();
            //var y = Rotation * x + Offset;
            var y = (Rotation * (x-SourceCentroid)) + Offset + SourceCentroid; //shift point to center, apply rotation, then shift to the destination
            var p = new PartLocation(y) {A = from.A + Angle};
            return p;
        }

        /// <summary>
        /// The RMS error of all the source to dest points
        /// </summary>
        /// <returns></returns>
        public double RMSError() {
            double rms_error = 0;
            for (int i = 0; i < source.Count; i++) {
                var b = Map(source[i]);
                rms_error += Math.Pow(b.DistanceTo(dest[i]), 2);
            }
            rms_error = Math.Sqrt(rms_error);
            return rms_error;
        }

        

        /// <summary>
        /// The furthest distance a fiducial moved
        /// </summary>
        /// <returns></returns>
        public double MaxFiducialMovement() {
            Global.Instance.DisplayText(String.Format("Offset = {0}  Angle = {1}", new PartLocation(Offset) - Global.Instance.Locations.GetLocation("PCB Zero") , Angle), System.Drawing.Color.Purple);
            List<double> distances = new List<double>();
            for (int i = 0; i < source.Count; i++) {
                var s = new PartLocation(source[i]);
                var d = new PartLocation(dest[i]);
                var m = Map(source[i]);
               // Global.Instance.DisplayText(String.Format("Source {0}  Dest {1}  Mapped {2}", s, d, m), System.Drawing.Color.Purple);
                distances.Add(Map(source[i]).DistanceTo(dest[i]));
            }
            return distances.Max();
        }
                

    }
}
