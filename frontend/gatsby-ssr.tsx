import type { GatsbySSR } from "gatsby";
import * as React from "react";

export const onRenderBody: GatsbySSR["onRenderBody"] = ({ setHeadComponents }) => {
  setHeadComponents([
    // Removed external font imports to revert to system fonts
    <link
      key="favicon-svg"
      rel="icon"
      type="image/svg+xml"
      href="/favicon.svg"
    />,
  ]);
};
