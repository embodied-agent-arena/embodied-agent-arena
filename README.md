# Embodied Agent Arena

**Are Frontier VLM Agents Ready to Be Robot Generalists?**  
*An Empirical Study with the Embodied Agent Arena*

[Project page](https://embodied-agent-arena.haojianhuang927.workers.dev/) · [Paper](https://embodied-agent-arena.haojianhuang927.workers.dev/paper.pdf)

Embodied Agent Arena evaluates seven frontier vision-language agents on 1,000 cases across Geometry, Spatial Reasoning, Affordance, Task Planning, and Manipulation. It combines 32 established sources with GeoProbe, a new 168-case geometric-estimation benchmark.

## Project page source

The static project page is in `public/`. Edit `index.html`, `styles.css`, and `app.js` to update the layout and content. Research figures are in `public/assets/`, and the paper is `public/paper.pdf`.

Preview locally:

```sh
python3 -m http.server 8000 --directory public
```

Open `http://localhost:8000`. Deploy to Cloudflare Workers with the included `wrangler.jsonc`:

```sh
npx wrangler deploy
```

## Authors

Haojian Huang, Pukun Zhao, Zexi Li, Yehang Zhang, Yangkai Wei, Wenqian Li, Han Yang, Kaiwen Zhou, Ying-Cong Chen, and Yinchuan Li.

HKUST (Guangzhou) · The Chinese University of Hong Kong · Knowin AI
