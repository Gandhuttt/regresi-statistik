import traceback, base64, io, os, csv
from flask import Flask, render_template, request
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    results = None
    length = 3  # baris
    k = 1  # kolom X
    y_vals = ["" for _ in range(length)]
    x_vals = [[""] * k for _ in range(length)]
    plot_url = None

    if request.method == "POST":
        try:
            length = int(request.form.get("length", length))
            k = int(request.form.get("k", k))

            # ----- baca semua input manual yang ada di form -----
            y_vals = []
            x_vals = []
            parse_ok = True
            for i in range(length):
                y_raw = request.form.get(f"y_{i}", "").strip()
                y_vals.append(y_raw)
                row_x = []
                for j in range(k):
                    x_raw = request.form.get(f"x_{i}_{j}", "").strip()
                    row_x.append(x_raw)
                    if not x_raw or not y_raw:
                        parse_ok = False
                    else:
                        try:
                            float(x_raw)
                            float(y_raw)
                        except ValueError:
                            parse_ok = False
                x_vals.append(row_x)

            action = request.form.get("action", "")

            # ================= CSV UPLOAD BRANCH =================
            if action == "upload_csv":
                file = request.files.get("csv_file")
                if not file or file.filename == "":
                    error = "Harap pilih file CSV terlebih dahulu."
                else:
                    try:
                        # Baca file CSV ke memori (string)
                        stream = io.StringIO(file.stream.read().decode("utf-8-sig"))
                        reader = csv.reader(stream)
                        rows = [row for row in reader if any(cell.strip() for cell in row)]

                        if not rows:
                            error = "File CSV kosong."
                        else:
                            def is_float(s: str) -> bool:
                                try:
                                    float(s)
                                    return True
                                except ValueError:
                                    return False

                            data_rows = rows

                            # Jika baris pertama mengandung non-numeric → anggap header
                            if any(not is_float(c) for c in rows[0]):
                                data_rows = rows[1:]

                            if not data_rows:
                                error = "Tidak ada data numerik di CSV."
                            else:
                                length = len(data_rows)
                                k = len(data_rows[0]) - 1  # 1 kolom Y + k kolom X
                                if k < 1:
                                    error = "CSV minimal harus punya 2 kolom (Y dan minimal 1 X)."
                                else:
                                    y_vals = []
                                    x_vals = []
                                    parse_ok_csv = True

                                    for r in data_rows:
                                        # Normalisasi panjang baris (pad jika kurang)
                                        if len(r) < k + 1:
                                            r = r + [""] * (k + 1 - len(r))

                                        y = r[0].strip()
                                        xs = [c.strip() for c in r[1:k+1]]

                                        if not y or any(not c for c in xs):
                                            parse_ok_csv = False
                                            break

                                        try:
                                            float(y)
                                            for c in xs:
                                                float(c)
                                        except ValueError:
                                            parse_ok_csv = False
                                            break

                                        y_vals.append(y)
                                        x_vals.append(xs)

                                    if not parse_ok_csv:
                                        error = "CSV mengandung nilai kosong atau non-numerik."

                    except Exception as e:
                        error = f"Gagal membaca CSV: {str(e)}"

                # Setelah upload CSV, cukup tampilkan tabel terisi.
                # User bisa edit lalu klik "Hitung Regresi".
                return render_template(
                    "index.html",
                    length=length,
                    k=k,
                    y_vals=y_vals,
                    x_vals=x_vals,
                    error=error,
                    results=None,
                    plot_url=None,
                )

            # ================= AKSI TAMBAH / HAPUS =================
            if action == "add_row":
                length += 1
                y_vals.append("")
                x_vals.append([""] * k)

            elif action == "add_col":
                k += 1
                for row in x_vals:
                    row.append("")
                if length == k + 1:
                    length += 1
                    y_vals.append("")
                    x_vals.append([""] * k)

            elif action.startswith("del_col"):
                col_idx = int(action.replace("del_col", ""))
                if k > 1 and 0 <= col_idx < k:
                    k -= 1
                    for row in x_vals:
                        if col_idx < len(row):
                            row.pop(col_idx)
                else:
                    error = "⚠️ Minimal harus tersisa 1 kolom X"

            elif action.startswith("del_row"):
                idx = int(action.replace("del_row", ""))
                if 0 <= idx < length:
                    y_vals.pop(idx)
                    x_vals.pop(idx)
                    length -= 1

            # ================= AKSI HITUNG REGRESI =================
            elif action == "submit":
                if length < 3:
                    error = "Minimal 3 baris data"
                elif not parse_ok:
                    error = "Semua field harus angka valid"
                elif length <= k + 1:
                    error = "⚠️ Jumlah baris harus > jumlah prediktor + 1 (minimal {} baris)".format(
                        k + 2
                    )
                else:
                    # ----- siapkan array numerik -----
                    Y = np.array([float(y) for y in y_vals])
                    X = np.array([[float(x) for x in row] for row in x_vals])
                    if np.any(np.std(X, axis=0) == 0):
                        error = "Salah satu kolom X konstan"
                    else:
                        reg = LinearRegression().fit(X, Y)
                        y_pred = reg.predict(X)
                        r2 = r2_score(Y, y_pred)
                        n, p = X.shape
                        resid = Y - y_pred
                        mse = np.sum(resid**2) / (n - p - 1) if (n - p - 1) > 0 else np.inf
                        
                        # Tambah intercept ke X untuk SE yang benar
                        X_with_intercept = np.column_stack([np.ones(n), X])
                        se_all = np.sqrt(
                            np.diag(np.linalg.inv(X_with_intercept.T @ X_with_intercept)) * mse
                        )
                        se_intercept = se_all[0]
                        se_coef = se_all[1:]
                        
                        # t-statistic & p-value intercept
                        t_intercept = reg.intercept_ / se_intercept
                        p_intercept = 2 * (1 - stats.t.cdf(np.abs(t_intercept), df=n - p - 1))
                        
                        # t-statistic & p-value slope
                        t_stats = reg.coef_ / se_coef
                        p_vals = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n - p - 1))

                        # ===== UJI-F =====
                        tss = np.sum((Y - np.mean(Y))**2)
                        rss = np.sum((y_pred - np.mean(Y))**2)
                        ess = np.sum(resid**2)

                        df_regression = p
                        df_residual = n - p - 1
                        
                        ms_regression = rss / df_regression
                        ms_residual = ess / df_residual
                        
                        f_statistic = ms_regression / ms_residual
                        f_pvalue = 1 - stats.f.cdf(f_statistic, df_regression, df_residual)

                        results = {
                            "intercept": f"{reg.intercept_:.6f}",
                            "slope": [f"{c:.6f}" for c in reg.coef_],
                            "r2": f"{r2:.6f}",
                            "tstat_intercept": f"{t_intercept:.6f}",
                            "pval_intercept": f"{p_intercept:.6f}",
                            "tstat": [f"{t:.6f}" for t in t_stats],
                            "pval": [f"{p:.6f}" for p in p_vals],
                            "f_statistic": f"{f_statistic:.6f}",
                            "f_pvalue": f"{f_pvalue:.6f}",
                            "df_regression": df_regression,
                            "df_residual": df_residual,
                            "ms_regression": f"{ms_regression:.6f}",
                            "ms_residual": f"{ms_residual:.6f}",
                        }

                        # ---------- BUAT GRAFIK ----------
                        if k > 1:
                            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                            axes = axes.flatten()

                            # 1. Actual vs Predicted
                            axes[0].scatter(Y, y_pred, alpha=0.6, edgecolors='k', s=80)
                            axes[0].plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'r--', lw=2, label='Perfect Fit')
                            axes[0].set_xlabel('Actual Y')
                            axes[0].set_ylabel('Predicted Y')
                            axes[0].set_title('Actual vs Predicted')
                            axes[0].legend()
                            axes[0].grid(True, alpha=0.3)

                            # 2. Residual plot
                            axes[1].scatter(y_pred, resid, alpha=0.6, edgecolors='k', s=80)
                            axes[1].axhline(0, color='r', ls='--', lw=2)
                            axes[1].set_xlabel('Predicted Y')
                            axes[1].set_ylabel('Residuals')
                            axes[1].set_title('Residual Plot')
                            axes[1].grid(True, alpha=0.3)

                            # 3. Q-Q plot
                            stats.probplot(resid, dist="norm", plot=axes[2])
                            axes[2].set_title('Q-Q Plot of Residuals')
                            axes[2].grid(True, alpha=0.3)

                            # 4. Partial regression plot
                            for j in range(k):
                                X_not_j = np.delete(X, j, axis=1)
                                res_y = Y - LinearRegression().fit(X_not_j, Y).predict(X_not_j)
                                res_xj = X[:, j] - LinearRegression().fit(X_not_j, X[:, j]).predict(X_not_j)
                                axes[3].scatter(res_xj, res_y, alpha=0.6, edgecolors='k', s=50, label=f'X{j+1}')
                            axes[3].axhline(0, color='k', lw=0.5)
                            axes[3].axvline(0, color='k', lw=0.5)
                            axes[3].set_xlabel('Residuals of Xj | others')
                            axes[3].set_ylabel('Residuals of Y | others')
                            axes[3].set_title('Partial Regression Plot')
                            axes[3].legend()
                            axes[3].grid(True, alpha=0.3)

                        else:
                            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

                            # 1. Actual vs Predicted
                            axes[0].scatter(Y, y_pred, alpha=0.6, edgecolors='k', s=80)
                            axes[0].plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'r--', lw=2, label='Perfect Fit')
                            axes[0].set_xlabel('Actual Y')
                            axes[0].set_ylabel('Predicted Y')
                            axes[0].set_title('Actual vs Predicted')
                            axes[0].legend()
                            axes[0].grid(True, alpha=0.3)

                            # 2. Residual plot
                            axes[1].scatter(y_pred, resid, alpha=0.6, edgecolors='k', s=80)
                            axes[1].axhline(0, color='r', ls='--', lw=2)
                            axes[1].set_xlabel('Predicted Y')
                            axes[1].set_ylabel('Residuals')
                            axes[1].set_title('Residual Plot')
                            axes[1].grid(True, alpha=0.3)

                            # 3. Q-Q plot
                            stats.probplot(resid, dist="norm", plot=axes[2])
                            axes[2].set_title('Q-Q Plot of Residuals')
                            axes[2].grid(True, alpha=0.3)

                        plt.tight_layout()

                        img = io.BytesIO()
                        plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
                        img.seek(0)
                        plot_url = base64.b64encode(img.getvalue()).decode()
                        plt.close()

        except Exception as e:
            error = f"Kesalahan: {str(e)}"
            traceback.print_exc()

    return render_template(
        "index.html",
        length=length,
        k=k,
        y_vals=y_vals,
        x_vals=x_vals,
        error=error,
        results=results,
        plot_url=plot_url,
    )


if __name__ == "__main__":
    app.run(debug=True)
