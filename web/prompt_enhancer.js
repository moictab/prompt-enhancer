import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

app.registerExtension({
    name: "Comfy.PromptEnhancer",

    async beforeRegisterNodeDef(nodeType, nodeData, _app) {
        if (nodeData.name !== "PromptEnhancer") return;

        // Add read-only text widgets when the node is first created
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);

            const pos = ComfyWidgets.STRING(
                this, "positive_output",
                ["STRING", { multiline: true }], app
            );
            pos.widget.inputEl.readOnly = true;
            pos.widget.inputEl.style.opacity = "0.8";
            pos.widget.inputEl.style.fontSize = "12px";

            const neg = ComfyWidgets.STRING(
                this, "negative_output",
                ["STRING", { multiline: true }], app
            );
            neg.widget.inputEl.readOnly = true;
            neg.widget.inputEl.style.opacity = "0.8";
            neg.widget.inputEl.style.fontSize = "12px";

            // Reasonable default size
            this.setSize([400, 520]);
        };

        // Update widgets when the node finishes executing
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);

            if (!message?.text) return;

            const posWidget = this.widgets?.find(w => w.name === "positive_output");
            const negWidget = this.widgets?.find(w => w.name === "negative_output");

            if (posWidget) posWidget.value = message.text[0] || "";
            if (negWidget) negWidget.value = message.text[1] || "";

            // Resize to fit content
            this.setSize(this.computeSize());
        };
    },
});
