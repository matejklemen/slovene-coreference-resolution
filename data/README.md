## SentiCoref file structure

```
[format]   
[layers]   
<empty_line>   
<empty_line>   
[body]   
```


**About [format]:**
- example: `#FORMAT=WebAnno TSV 3.2`

**About [layers]:**
- marked by the # character followed by
  - `T_SP`=span types
  - `T_CH`=chain layers
  - `T_RL`=relation layers
- every layer is written in new line, followed by the features in the layer
- features are separated by the | character and only the short name of the feature is provided
- slot features start with a prefix ROLE_ followed by the name of the role and the link

**About [body]:**
- Reserved characters have a special meaning in the TSV format and must be are escaped with the backslash
  (\) character if they appear in text or feature values. Reserved characters are the following:
  \,[,],|,_,->,;,\t,\n,*
- Sentence annotations are presented following the text marker #Text=, before the token annotations
- If a sentence spans multiple lines, the text is split and multiple #Text= lines are generated
- Line format:
    sentence-token<TAB>begin_token_offset-end_token_offset<TAB>token
- line format notes:
  - Sub-token representations are affixed with a . and a number starts from 1 to N. Example:   
        ```
        1-3	9-14	plays     
        1-3.1	9-13	play     
        1-3.2	13-14	s     
        ```
  - For every features of a span Annotation, annotation value will be presented in the same row as
      the token/sub-token annotation, separated by a TAB character. If there is no annotation for the
      given span layer, a _ character is placed in the column. If the feature has no/null annotation or
      if the span layer do not have a feature at all, a * character represents the annotation.
- Chain layers will have always two features, referenceType and referenceRelation.
- Todo: more if needed