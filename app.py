
import joblib
import pandas as pd
import streamlit as st

from pymatgen.core import Composition, Lattice


st.set_page_config(
    page_title="B-O Band Gap Predictor",
    layout="centered"
)

AMU_PER_A3_TO_G_PER_CM3 = 1.66053906660


@st.cache_resource
def load_model():
    return joblib.load("BO_bandgap_model.joblib")


bundle = load_model()

model = bundle["model"]
featurizer = bundle["composition_featurizer"]
selected_columns = bundle["selected_feature_columns"]


def predict_band_gap(
    formula,
    number_of_sites,
    a, b, c,
    alpha, beta, gamma,
    space_group_number
):
    composition = Composition(formula)

    elements = {
        str(element)
        for element in composition.elements
    }

    if not {"B", "O"}.issubset(elements):
        raise ValueError("化学式必须同时含有B和O")

    lattice = Lattice.from_parameters(
        a=a,
        b=b,
        c=c,
        alpha=alpha,
        beta=beta,
        gamma=gamma
    )

    volume = lattice.volume
    volume_per_atom = volume / number_of_sites

    formula_units = (
        number_of_sites
        / composition.num_atoms
    )

    cell_mass = (
        float(composition.weight)
        * formula_units
    )

    density = (
        cell_mass
        * AMU_PER_A3_TO_G_PER_CM3
        / volume
    )

    input_df = pd.DataFrame({
        "composition": [composition]
    })

    features = featurizer.featurize_dataframe(
        input_df,
        col_id="composition",
        ignore_errors=False,
        pbar=False
    )

    structure_values = {
        "number_of_elements":
            len(composition.elements),
        "number_of_sites":
            number_of_sites,
        "density_g_cm3":
            density,
        "volume_per_atom_A3":
            volume_per_atom,
        "lattice_a_A": a,
        "lattice_b_A": b,
        "lattice_c_A": c,
        "lattice_alpha_degree": alpha,
        "lattice_beta_degree": beta,
        "lattice_gamma_degree": gamma,
        "space_group_number":
            space_group_number
    }

    for column, value in structure_values.items():
        features[column] = value

    model_input = features[selected_columns]

    prediction = model.predict(model_input)[0]
    prediction = max(0.0, float(prediction))

    return prediction, density, volume_per_atom


st.title("B-O Band Gap Predictor")
st.caption("适用于同时含有B和O的材料")

formula = st.text_input(
    "Chemical formula",
    value="Ag3BO3"
)

number_of_sites = st.number_input(
    "Number of sites",
    min_value=1,
    value=7,
    step=1
)

st.subheader("Lattice lengths")

col1, col2, col3 = st.columns(3)

with col1:
    a = st.number_input("a (Å)", value=5.95966)

with col2:
    b = st.number_input("b (Å)", value=5.95966)

with col3:
    c = st.number_input("c (Å)", value=5.95966)

st.subheader("Lattice angles")

col4, col5, col6 = st.columns(3)

with col4:
    alpha = st.number_input(
        "α (°)",
        value=116.59028
    )

with col5:
    beta = st.number_input(
        "β (°)",
        value=116.59028
    )

with col6:
    gamma = st.number_input(
        "γ (°)",
        value=116.59027
    )

space_group_number = st.number_input(
    "Space group number",
    min_value=1,
    max_value=230,
    value=155,
    step=1
)

if st.button(
    "Predict Band Gap",
    type="primary",
    use_container_width=True
):
    try:
        prediction, density, volume_per_atom = (
            predict_band_gap(
                formula=formula,
                number_of_sites=number_of_sites,
                a=a,
                b=b,
                c=c,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                space_group_number=
                    space_group_number
            )
        )

        st.success("Prediction completed")

        st.metric(
            "Predicted Band Gap",
            f"{prediction:.3f} eV"
        )

        col7, col8 = st.columns(2)

        col7.metric(
            "Calculated Density",
            f"{density:.3f} g/cm³"
        )

        col8.metric(
            "Volume per Atom",
            f"{volume_per_atom:.3f} Å³"
        )

    except Exception as error:
        st.error(str(error))
