import React, { useState } from "react";
import ModalDialogScrollable from "./ModalDialogScrollable";

function Prediction() {
  const [showModal, setShowModal] = useState(false);

  const handlePredictClick = () => {
    setShowModal(true);
  };

  return (
    <div>
      <button
        type="button"
        onClick={handlePredictClick}
      >
        Predict
      </button>
      <ModalDialogScrollable showModal={showModal} setShowModal={setShowModal} />
    </div>
  );
}

export default Prediction;
